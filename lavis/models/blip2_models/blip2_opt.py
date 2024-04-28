"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging
from packaging import version

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import torch.nn.functional as F

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, Blip2ProteinBase, disabled_train
# from lavis.models.blip2_models.modeling_opt import OPTForCausalLM, OPTConfig
from transformers import AutoTokenizer, OPTForCausalLM, OPTConfig, LlamaTokenizer, MistralForCausalLM
import transformers
import esm
import random


def comb(s):
    s_list = [i.strip() for i in s.split(';')]
    random.shuffle(s_list)
    return '; '.join(s_list)

def process_text(txts, probs):
    res = dict()
    for txt, prob in zip(txts, probs):
        txt_sep = [x.strip() for x in txt.split(';')]
        for txt_sub in txt_sep:
            txt_sub = txt_sub.replace('|', '')
            if txt_sub not in res and txt_sub != '':
                res[txt_sub] = round(prob.item(),3)
    return '; '.join([str((k, v)) for k, v in res.items()])


@registry.register_model("blip2_opt")
class Blip2OPT(Blip2Base):
    """
    BLIP2 OPT model.
    Supported model types:
        - pretrained_opt2.7b: pretrained model with OPT2.7b
        - pretrained_opt6.7b: pretrained model with OPT6.7b
        - caption_coco_opt2.7b: fintuned image captioning model with OPT2.7b
        - caption_coco_opt6.7b: fintuned image captioning model with OPT6.7b
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_opt", "caption_coco_opt2.7b")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_opt2.7b": "configs/models/blip2/blip2_pretrain_opt2.7b.yaml",
        "pretrain_opt6.7b": "configs/models/blip2/blip2_pretrain_opt6.7b.yaml",
        "caption_coco_opt2.7b": "configs/models/blip2/blip2_caption_opt2.7b.yaml",
        "caption_coco_opt6.7b": "configs/models/blip2/blip2_caption_opt6.7b.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        opt_model="facebook/opt-2.7b",
        prompt="",
        max_txt_len=32,
        apply_lemmatizer=False,
    ):
        """
        apply_lemmatizer: when set to True, postprocess predict_answers() result with lemmas.
        """
        super().__init__()
        transformers_version = version.parse(transformers.__version__)
        assert transformers_version >= version.parse("4.27"), "BLIP-2 OPT requires transformers>=4.27"
        
        self.tokenizer = self.init_tokenizer()

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")

        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        self.opt_tokenizer = AutoTokenizer.from_pretrained(opt_model, use_fast=False)
        self.opt_model = OPTForCausalLM.from_pretrained(
            opt_model, torch_dtype=torch.float16
        )
        for name, param in self.opt_model.named_parameters():
            param.requires_grad = False
        self.eos_token_id = self.opt_tokenizer(
            "\n", add_special_tokens=False
        ).input_ids[0]

        self.opt_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.opt_model.config.hidden_size
        )

        self.max_txt_len = max_txt_len
        self.prompt = prompt
        prompt_tokens = self.opt_tokenizer(self.prompt, return_tensors="pt")
        self.prompt_length = prompt_tokens.attention_mask.sum(1)
        
        self._apply_lemmatizer = apply_lemmatizer
        self._lemmatizer = None       

    def forward(self, samples):
        image = samples["image"]
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        inputs_opt = self.opt_proj(query_output.last_hidden_state)
        atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(image.device)

        self.opt_tokenizer.padding_side = "right"

        text = [t + "\n" for t in samples["text_input"]]

        opt_tokens = self.opt_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(image.device)

        targets = opt_tokens.input_ids.masked_fill(
            opt_tokens.input_ids == self.opt_tokenizer.pad_token_id, -100
        )
        if self.prompt:
            targets[:, : self.prompt_length] = -100  # do not apply loss to the prompt

        empty_targets = (
            torch.ones(atts_opt.size(), dtype=torch.long).to(image.device).fill_(-100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        inputs_embeds = self.opt_model.model.decoder.embed_tokens(opt_tokens.input_ids)
        inputs_embeds = torch.cat([inputs_opt, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)

        with self.maybe_autocast():
            outputs = self.opt_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        loss = outputs.loss

        return {"loss": loss}

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=30,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        image = samples["image"]
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                image.device
            )

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            inputs_opt = self.opt_proj(query_output.last_hidden_state)
            atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(
                image.device
            )

            if "prompt" in samples.keys():
                prompt = samples["prompt"]
            else:
                prompt = self.prompt

            prompt = [prompt] * image.size(0)

            opt_tokens = self.opt_tokenizer(
                prompt,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
            ).to(image.device)
            attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)
            
            # new version for transformers>=4.27
            inputs_embeds = self.opt_model.get_input_embeddings()(opt_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_opt,inputs_embeds],dim=1)
            
            outputs = self.opt_model.generate(
                inputs_embeds=inputs_embeds, 
                attention_mask=attention_mask,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                eos_token_id=self.eos_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )
            output_text = self.opt_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )
                            
            # previous version for transformers<4.27
            # if use_nucleus_sampling:
            #     query_embeds = inputs_opt.repeat_interleave(num_captions, dim=0)
            #     num_beams = 1
            # else:
            #     query_embeds = inputs_opt.repeat_interleave(num_beams, dim=0)

            # outputs = self.opt_model.generate(
            #     input_ids=input_ids,
            #     query_embeds=query_embeds,
            #     attention_mask=attention_mask,
            #     do_sample=use_nucleus_sampling,
            #     top_p=top_p,
            #     temperature=temperature,
            #     num_beams=num_beams,
            #     max_new_tokens=max_length,
            #     min_length=min_length,
            #     eos_token_id=self.eos_token_id,
            #     repetition_penalty=repetition_penalty,
            #     length_penalty=length_penalty,
            #     num_return_sequences=num_captions,
            # )

            # prompt_length = opt_tokens.input_ids.shape[1]
            # output_text = self.opt_tokenizer.batch_decode(
            #     outputs[:, prompt_length:], skip_special_tokens=True
            # )
            
            output_text = [text.strip() for text in output_text]
            return output_text
        
        
    def predict_answers(
        self,
        samples,
        num_beams=5,
        inference_method="generate",
        max_len=10,
        min_len=1,
        num_ans_candidates=128,
        answer_list=None,
        prompt="",
        length_penalty=0,
        **kwargs
    ):
        image = samples["image"]
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                image.device
            )

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            inputs_opt = self.opt_proj(query_output.last_hidden_state)
            atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(
                image.device
            )

            if isinstance(samples["text_input"], str):
                samples["text_input"] = [samples["text_input"]]
            if prompt:
                text_input = [prompt.format(question) for question in samples["text_input"]]
            else:
                text_input = samples["text_input"]

            self.opt_tokenizer.padding_side = "left"
            opt_tokens = self.opt_tokenizer(
                text_input,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
            ).to(image.device)
        
            attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)
            
            # require transformers>=4.27
            inputs_embeds = self.opt_model.get_input_embeddings()(opt_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_opt,inputs_embeds],dim=1)
            
            outputs = self.opt_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=False,
                num_beams=num_beams,
                max_new_tokens=max_len,
                min_length=min_len,
                eos_token_id=self.eos_token_id,
                length_penalty=length_penalty,
            )
            output_text = self.opt_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )
            output_text = [text.strip() for text in output_text]
        if self._apply_lemmatizer or ("apply_lemmatizer" in samples.keys() and samples["apply_lemmatizer"]):
            output_text = self._lemmatize(output_text)

        return output_text
    
    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer
        
    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        opt_model = cfg.get("opt_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 32)
        
        apply_lemmatizer = cfg.get("apply_lemmatizer", False)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            opt_model=opt_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
            apply_lemmatizer=apply_lemmatizer,
        )
        model.load_checkpoint_from_config(cfg)

        return model


@registry.register_model("blip2_protein_opt")
class Blip2ProteinOPT(Blip2ProteinBase):

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_protein_opt350m": "configs/models/blip2/pretrain_protein_opt350m.yaml",
        "pretrain_protein_opt2.7b": "configs/models/blip2/pretrain_protein_opt2.7b.yaml",
    }

    def __init__(
            self,
            freeze_vit=True,
            num_query_token=32,
            opt_model="facebook/opt-350m",
            prompt="",
            max_txt_len=128,
            max_protein_len=128,
            apply_lemmatizer=False,
            get_eval=False,
            esm_size='650m'
    ):
        """
        apply_lemmatizer: when set to True, postprocess predict_answers() result with lemmas.
        """
        super().__init__()
        transformers_version = version.parse(transformers.__version__)
        assert transformers_version >= version.parse("4.27"), "BLIP-2 OPT requires transformers>=4.27"

        self.tokenizer = self.init_tokenizer()
        '''
        self.ln_vision, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        if freeze_vit:
            self.ln_vision = self.ln_vision.half()
        self.visual_encoder = alphabet.get_batch_converter(truncation_seq_length=max_protein_len)
        self.padding_idx = alphabet.padding_idx
        self.vis_layers = self.ln_vision.num_layers

        if freeze_vit:
            for name, param in self.ln_vision.named_parameters():
                param.requires_grad = False
            self.ln_vision = self.ln_vision.eval()
            self.ln_vision.train = disabled_train
            logging.info("freeze vision encoder")
        else:
            for name, param in self.ln_vision.named_parameters():
                if 'contact_head' in name or 'emb_layer_norm_after' in name or 'lm_head' in name:
                    param.requires_grad = False
        '''
        if esm_size == '650m':
            self.Qformer, self.query_tokens = self.init_Qformer(num_query_token, 1280)
        elif esm_size == '3b':
            self.Qformer, self.query_tokens = self.init_Qformer(num_query_token, 2560)
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        #self.opt_tokenizer = AutoTokenizer.from_pretrained(opt_model, use_fast=False)
        #self.opt_model = OPTForCausalLM.from_pretrained(
        #    opt_model, torch_dtype=torch.float16
        #)
        self.opt_tokenizer = LlamaTokenizer.from_pretrained("/cluster/home/wenkai/.cache/huggingface/hub/models--teknium--OpenHermes-2.5-Mistral-7B", use_fast=False)
        self.opt_tokenizer.pad_token = '<pad>'
        if get_eval:
            self.opt_model = MistralForCausalLM.from_pretrained("/cluster/home/wenkai/.cache/huggingface/hub/models--teknium--OpenHermes-2.5-Mistral-7B", torch_dtype=torch.float16)
            for name, param in self.opt_model.named_parameters():
                param.requires_grad = False
        else:
            self.opt_model = MistralForCausalLM.from_pretrained("/cluster/home/wenkai/.cache/huggingface/hub/models--teknium--OpenHermes-2.5-Mistral-7B", torch_dtype=torch.float16)
            print(self.opt_model)                
            for name, param in self.opt_model.named_parameters():
                param.requires_grad = False
            #self.opt_model.lm_head = self.opt_model.lm_head.float()
            #for param in self.opt_model.lm_head.parameters():
            #    param.requires_grad = True

        #self.eos_token_id = self.opt_tokenizer(
        #    "\n", add_special_tokens=False
        #).input_ids[0]
        self.eos_token_id = self.opt_tokenizer(
            "\n", add_special_tokens=False
        ).input_ids[1]
        print(f"LLM hidden size: {self.opt_model.config.hidden_size}")
        self.opt_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.opt_model.config.hidden_size
        )

        self.max_txt_len = max_txt_len
        self.prompt = prompt
        prompt_tokens = self.opt_tokenizer(self.prompt, return_tensors="pt")
        self.prompt_length = prompt_tokens.attention_mask.sum(1)

        self._apply_lemmatizer = apply_lemmatizer
        self._lemmatizer = None
        self.get_eval = get_eval

    def forward(self, samples):
        '''
        image = samples["image"]
        image = [('protein{}'.format(i), x) for i, x in enumerate(image)]

        with self.maybe_autocast():
            _, _, batch_tokens = self.visual_encoder(image)
            image_embeds = self.ln_vision(batch_tokens.to(self.device), repr_layers=[self.vis_layers], return_contacts=True)["representations"][self.vis_layers].contiguous()
        '''
        image_embeds = samples["image"]
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            self.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        inputs_opt = self.opt_proj(query_output.last_hidden_state)

        #torch.save(query_output.last_hidden_state, '/cluster/home/wenkai/LAVIS/output/mf_bp_cc/query_output_mf/{}.pt'.format(samples['name'][0]))
        #torch.save(inputs_opt, '/cluster/home/wenkai/LAVIS/output/mf_bp_cc/inputs_opt_mf/{}.pt'.format(samples['name'][0]))

        atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(self.device)

        # prompt
        prompt = samples["prompt"]
        prompt_tokens = self.opt_tokenizer(prompt, padding="longest", return_tensors="pt")
        prompt_length = prompt_tokens.attention_mask.sum(1)

        self.opt_tokenizer.padding_side = "right"

        text = [p+' '+comb(t) + "\n" for p, t in zip(prompt, samples["text_input"])]
        text = [p+' '+ t + "\n" for p, t in zip(prompt, samples["text_input"])]

        opt_tokens = self.opt_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(self.device)

        targets = opt_tokens.input_ids.masked_fill(
            opt_tokens.input_ids == self.opt_tokenizer.pad_token_id, -100
        )

        for i, pl in enumerate(prompt_length):
            targets[i, :pl] = -100  # do not apply loss to the prompt
        #print(prompt_tokens, '\n', opt_tokens, '\n', prompt_length)

	#if self.prompt:
        #    targets[:, : self.prompt_length] = -100  # do not apply loss to the prompt

        empty_targets = (
            torch.ones(atts_opt.size(), dtype=torch.long).to(self.device).fill_(-100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        #inputs_embeds = self.opt_model.model.decoder.embed_tokens(opt_tokens.input_ids)
        inputs_embeds = self.opt_model.model.embed_tokens(opt_tokens.input_ids)
        inputs_embeds = torch.cat([inputs_opt, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)
        
        with self.maybe_autocast():
            outputs = self.opt_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        loss = outputs.loss
        if self.get_eval:
            label = samples["text_input"]
            name = samples['name']
            text = samples['prompt']
            #text = ['' for i in range(len(label))]
            opt_tokens = self.opt_tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
            ).to(self.device)
            #inputs_embeds = self.opt_model.model.decoder.embed_tokens(opt_tokens.input_ids)
            inputs_embeds = self.opt_model.model.embed_tokens(opt_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_opt, inputs_embeds], dim=1)
            attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)
            #if name[0] == 'Pin':
            #    torch.save(inputs_embeds, '/cluster/home/wenkai/LAVIS/output/inputs_embeds.pt')
            #    torch.save(attention_mask, '/cluster/home/wenkai/LAVIS/output/attention_mask.pt')
            
            #self.get_eval = False
            #'''
            num_txt = 15
            return_num_txt = 10
            with torch.no_grad():
                outputs = self.opt_model.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, min_length=1,
                                                  max_length=32,temperature=1.,return_dict_in_generate=True, output_scores=True,
                                                  repetition_penalty=1., num_beams=num_txt,
                                                  length_penalty=0.2, num_return_sequences=return_num_txt,eos_token_id=self.eos_token_id)
            output_text = self.opt_tokenizer.batch_decode(outputs['sequences'], skip_special_tokens=True)
            '''
            num_txt = 5
            return_num_txt = 1
            with torch.no_grad():
                outputs = self.opt_model.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, min_length=1,
                                                  max_length=128,temperature=1.,return_dict_in_generate=True, output_scores=True,
                                                  repetition_penalty=1., num_beams=num_txt,
                                                  length_penalty=1, num_return_sequences=return_num_txt,eos_token_id=self.eos_token_id)
            output_text = self.opt_tokenizer.batch_decode(outputs['sequences'], skip_special_tokens=True)
            '''
            probs = F.softmax(outputs['sequences_scores'])
            #print(output_text)
            output_text = [x.replace('\n', '').strip() for x in output_text]
            
            output_text_ = []
            for i in range(len(label)):
                #output_text_.append(';'.join(output_text[i*return_num_txt:(i+1)*return_num_txt]))
                output_text_.append(process_text(output_text[i*return_num_txt:(i+1)*return_num_txt], probs[i*return_num_txt:(i+1)*return_num_txt]))
            #output_text_ = ['; '.join(list(set([i.strip() for i in x.split(';')]))) for x in output_text_]
            #with open('/cluster/home/wenkai/LAVIS/output/mf_bp_cc/output_train_bp_exp_491966.txt', 'a+',  encoding="utf-8") as f:
            with open('/cluster/home/wenkai/LAVIS/output/mf_bp_cc/output_test_bp_cases_526432.txt', 'a+',  encoding="utf-8") as f:
                for i in range(len(label)):
                    f.write(name[i] + "|" +output_text_[i]+"|"+label[i]+'\n')
        return {"loss": loss}

    @torch.no_grad()
    def generate(
            self,
            samples,
            use_nucleus_sampling=False,
            num_beams=5,
            max_length=30,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.5,
            length_penalty=1.0,
            num_captions=1,
            temperature=1,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        with self.maybe_autocast():
            image_embeds = samples["image"]
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                self.device
            )
    
            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
    
            inputs_opt = self.opt_proj(query_output.last_hidden_state)
            atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(self.device)
    
            label = samples["text_input"]
            name = samples['name']
            text = samples['prompt']
            # text = ['' for i in range(len(label))]
            opt_tokens = self.opt_tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
            ).to(self.device)
            # inputs_embeds = self.opt_model.model.decoder.embed_tokens(opt_tokens.input_ids)
            inputs_embeds = self.opt_model.model.embed_tokens(opt_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_opt, inputs_embeds], dim=1)
            attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)
            # if name[0] == 'Pin':
            #    torch.save(inputs_embeds, '/cluster/home/wenkai/LAVIS/output/inputs_embeds.pt')
            #    torch.save(attention_mask, '/cluster/home/wenkai/LAVIS/output/attention_mask.pt')
    
            # self.get_eval = False
            #'''
            #num_txt = 15
            #return_num_txt = 10
            num_txt = 15
            return_num_txt = 10
            with torch.no_grad():
                outputs = self.opt_model.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, min_length=1,
                                                  max_length=32, temperature=1., return_dict_in_generate=True,
                                                  output_scores=True,
                                                  repetition_penalty=1., num_beams=num_txt,
                                                  length_penalty=0., num_return_sequences=return_num_txt,
                                                  eos_token_id=self.eos_token_id)
            output_text = self.opt_tokenizer.batch_decode(outputs['sequences'], skip_special_tokens=True)
            '''
            num_txt = 5
            return_num_txt = 1
            with torch.no_grad():
                outputs = self.opt_model.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, min_length=1,
                                                  max_length=96,temperature=1.,return_dict_in_generate=True, output_scores=True,
                                                  repetition_penalty=1., num_beams=num_txt,
                                                  length_penalty=1, num_return_sequences=return_num_txt,eos_token_id=self.eos_token_id)
            output_text = self.opt_tokenizer.batch_decode(outputs['sequences'], skip_special_tokens=True)
            '''
            probs = F.softmax(outputs['sequences_scores'])
            # print(output_text)
            output_text = [x.replace('\n', '').strip() for x in output_text]
    
            output_text_ = []
            for i in range(len(label)):
                # output_text_.append(';'.join(output_text[i*return_num_txt:(i+1)*return_num_txt]))
                output_text_.append(process_text(output_text[i * return_num_txt:(i + 1) * return_num_txt],
                                                 probs[i * return_num_txt:(i + 1) * return_num_txt]))
            #output_text_ = ['; '.join(list(set([i.strip() for i in x.split(';')]))) for x in output_text_]
            with open('/cluster/home/wenkai/LAVIS/output/mf_bp_cc/output_test_mf_exp_493552.txt', 'a+',  encoding="utf-8") as f:
                for i in range(len(label)):
                    f.write(name[i] + "|" +output_text_[i]+"|"+label[i]+'\n')
            return output_text_


    def predict_answers(
            self,
            samples,
            num_beams=5,
            inference_method="generate",
            max_len=10,
            min_len=1,
            num_ans_candidates=128,
            answer_list=None,
            prompt="",
            length_penalty=0,
            **kwargs
    ):
        image = samples["image"]
        image = [('protein{}'.format(i), x) for i, x in enumerate(image)]
        with self.maybe_autocast():
            _, _, batch_tokens = self.visual_encoder(image)
            image_embeds = self.ln_vision(batch_tokens.to(self.device), repr_layers=[self.vis_layers], return_contacts=True)["representations"][self.vis_layers].contiguous()

            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                self.device
            )

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            inputs_opt = self.opt_proj(query_output.last_hidden_state)
            atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(
                self.device
            )

            if isinstance(samples["text_input"], str):
                samples["text_input"] = [samples["text_input"]]
            if prompt:
                text_input = [prompt.format(question) for question in samples["text_input"]]
            else:
                text_input = samples["text_input"]

            self.opt_tokenizer.padding_side = "left"
            opt_tokens = self.opt_tokenizer(
                text_input,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
            ).to(self.device)

            attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)

            # require transformers>=4.27
            inputs_embeds = self.opt_model.get_input_embeddings()(opt_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_opt, inputs_embeds], dim=1)

            outputs = self.opt_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=False,
                num_beams=num_beams,
                max_new_tokens=max_len,
                min_length=min_len,
                eos_token_id=self.eos_token_id,
                length_penalty=length_penalty,
            )
            output_text = self.opt_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )
            output_text = [text.strip() for text in output_text]
        if self._apply_lemmatizer or ("apply_lemmatizer" in samples.keys() and samples["apply_lemmatizer"]):
            output_text = self._lemmatize(output_text)

        return output_text

    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer

    @classmethod
    def from_config(cls, cfg):
        num_query_token = cfg.get("num_query_token")
        opt_model = cfg.get("opt_model")

        freeze_vit = cfg.get("freeze_vit", True)
        get_eval = cfg.get("get_eval", False)
        esm_size = cfg.get("esm_size", '650m')
        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 128)
        max_protein_len = cfg.get("max_protein_len", 128)

        apply_lemmatizer = cfg.get("apply_lemmatizer", False)

        model = cls(
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            opt_model=opt_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
            max_protein_len=max_protein_len,
            apply_lemmatizer=apply_lemmatizer,
            get_eval=get_eval,
            esm_size=esm_size,
        )
        model.load_checkpoint_from_config(cfg)

        return model
