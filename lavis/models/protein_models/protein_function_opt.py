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
from transformers import AutoTokenizer, LlamaTokenizer, MistralForCausalLM
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



@registry.register_model("blip2_protein_mistral")
class Blip2ProteinMistral(Blip2ProteinBase):

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_protein_mistral7b": "configs/models/blip2/pretrain_protein_mistral7b.yaml",
    }

    def __init__(
            self,
            num_query_token=32,
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
        assert transformers_version >= version.parse("4.27"), "BLIP-2 mistral requires transformers>=4.27"

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

        self.mistral_tokenizer = LlamaTokenizer.from_pretrained("/cluster/home/wenkai/.cache/huggingface/hub/models--teknium--OpenHermes-2.5-Mistral-7B", use_fast=False)
        self.mistral_tokenizer.pad_token = '<pad>'
        if get_eval:
            self.mistral_model = MistralForCausalLM.from_pretrained("/cluster/home/wenkai/.cache/huggingface/hub/models--teknium--OpenHermes-2.5-Mistral-7B", torch_dtype=torch.float16)
            for name, param in self.mistral_model.named_parameters():
                param.requires_grad = False
        else:
            self.mistral_model = MistralForCausalLM.from_pretrained("/cluster/home/wenkai/.cache/huggingface/hub/models--teknium--OpenHermes-2.5-Mistral-7B", torch_dtype=torch.float16)
            print(self.mistral_model)
            for name, param in self.mistral_model.named_parameters():
                param.requires_grad = False
            #self.mistral_model.lm_head = self.mistral_model.lm_head.float()
            #for param in self.mistral_model.lm_head.parameters():
            #    param.requires_grad = True

        #self.eos_token_id = self.mistral_tokenizer(
        #    "\n", add_special_tokens=False
        #).input_ids[0]
        self.eos_token_id = self.mistral_tokenizer(
            "\n", add_special_tokens=False
        ).input_ids[1]
        print(f"LLM hidden size: {self.mistral_model.config.hidden_size}")
        self.mistral_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.mistral_model.config.hidden_size
        )

        self.max_txt_len = max_txt_len
        self.prompt = prompt
        prompt_tokens = self.mistral_tokenizer(self.prompt, return_tensors="pt")
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

        inputs_mistral = self.mistral_proj(query_output.last_hidden_state)

        #torch.save(query_output.last_hidden_state, '/cluster/home/wenkai/LAVIS/output/mf_bp_cc/query_output_mf/{}.pt'.format(samples['name'][0]))
        #torch.save(inputs_mistral, '/cluster/home/wenkai/LAVIS/output/mf_bp_cc/inputs_mistral_mf/{}.pt'.format(samples['name'][0]))

        atts_mistral = torch.ones(inputs_mistral.size()[:-1], dtype=torch.long).to(self.device)

        # prompt
        prompt = samples["prompt"]
        prompt_tokens = self.mistral_tokenizer(prompt, padding="longest", return_tensors="pt")
        prompt_length = prompt_tokens.attention_mask.sum(1)

        self.mistral_tokenizer.padding_side = "right"

        text = [p+' '+comb(t) + "\n" for p, t in zip(prompt, samples["text_input"])]
        text = [p+' '+ t + "\n" for p, t in zip(prompt, samples["text_input"])]

        mistral_tokens = self.mistral_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(self.device)

        targets = mistral_tokens.input_ids.masked_fill(
            mistral_tokens.input_ids == self.mistral_tokenizer.pad_token_id, -100
        )

        for i, pl in enumerate(prompt_length):
            targets[i, :pl] = -100  # do not apply loss to the prompt
        #print(prompt_tokens, '\n', mistral_tokens, '\n', prompt_length)

	#if self.prompt:
        #    targets[:, : self.prompt_length] = -100  # do not apply loss to the prompt

        empty_targets = (
            torch.ones(atts_mistral.size(), dtype=torch.long).to(self.device).fill_(-100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        #inputs_embeds = self.mistral_model.model.decoder.embed_tokens(mistral_tokens.input_ids)
        inputs_embeds = self.mistral_model.model.embed_tokens(mistral_tokens.input_ids)
        inputs_embeds = torch.cat([inputs_mistral, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_mistral, mistral_tokens.attention_mask], dim=1)

        with self.maybe_autocast():
            outputs = self.mistral_model(
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
            mistral_tokens = self.mistral_tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
            ).to(self.device)
            #inputs_embeds = self.mistral_model.model.decoder.embed_tokens(mistral_tokens.input_ids)
            inputs_embeds = self.mistral_model.model.embed_tokens(mistral_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_mistral, inputs_embeds], dim=1)
            attention_mask = torch.cat([atts_mistral, mistral_tokens.attention_mask], dim=1)
            #if name[0] == 'Pin':
            #    torch.save(inputs_embeds, '/cluster/home/wenkai/LAVIS/output/inputs_embeds.pt')
            #    torch.save(attention_mask, '/cluster/home/wenkai/LAVIS/output/attention_mask.pt')

            #self.get_eval = False
            #'''
            num_txt = 15
            return_num_txt = 10
            with torch.no_grad():
                outputs = self.mistral_model.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, min_length=1,
                                                  max_length=32,temperature=1.,return_dict_in_generate=True, output_scores=True,
                                                  repetition_penalty=1., num_beams=num_txt,
                                                  length_penalty=0.2, num_return_sequences=return_num_txt,eos_token_id=self.eos_token_id)
            output_text = self.mistral_tokenizer.batch_decode(outputs['sequences'], skip_special_tokens=True)
            '''
            num_txt = 5
            return_num_txt = 1
            with torch.no_grad():
                outputs = self.mistral_model.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, min_length=1,
                                                  max_length=128,temperature=1.,return_dict_in_generate=True, output_scores=True,
                                                  repetition_penalty=1., num_beams=num_txt,
                                                  length_penalty=1, num_return_sequences=return_num_txt,eos_token_id=self.eos_token_id)
            output_text = self.mistral_tokenizer.batch_decode(outputs['sequences'], skip_special_tokens=True)
            '''
            probs = F.softmax(outputs['sequences_scores'])
            #print(output_text)
            output_text = [x.replace('\n', '').strip() for x in output_text]
            
            output_text_ = []
            for i in range(len(label)):
                #output_text_.append(';'.join(output_text[i*return_num_txt:(i+1)*return_num_txt]))
                output_text_.append(process_text(output_text[i*return_num_txt:(i+1)*return_num_txt], probs[i*return_num_txt:(i+1)*return_num_txt]))
            #output_text_ = ['; '.join(list(set([i.strip() for i in x.split(';')]))) for x in output_text_]
            with open('/cluster/home/wenkai/LAVIS/output/mf_bp_cc/output_test_bp_cases_526432.txt', 'a+',  encoding="utf-8") as f:
                for i in range(len(label)):
                    f.write(name[i] + "|" +output_text_[i]+"|"+label[i]+'\n')
        return {"loss": loss}

    @torch.no_grad()
    def generate(
            self,
            samples,
            # use_nucleus_sampling=False,
            num_beams=15,
            max_length=32,
            min_length=1,
            # top_p=0.9,
            repetition_penalty=1.0,
            length_penalty=0.,
            num_captions=10,
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

            inputs_mistral = self.mistral_proj(query_output.last_hidden_state)
            atts_mistral = torch.ones(inputs_mistral.size()[:-1], dtype=torch.long).to(self.device)

            label = samples["text_input"]
            name = samples['name']
            text = samples['prompt']
            # text = ['' for i in range(len(label))]
            mistral_tokens = self.mistral_tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
            ).to(self.device)
            # inputs_embeds = self.mistral_model.model.decoder.embed_tokens(mistral_tokens.input_ids)
            inputs_embeds = self.mistral_model.model.embed_tokens(mistral_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_mistral, inputs_embeds], dim=1)
            attention_mask = torch.cat([atts_mistral, mistral_tokens.attention_mask], dim=1)
            # if name[0] == 'Pin':
            #    torch.save(inputs_embeds, '/cluster/home/wenkai/LAVIS/output/inputs_embeds.pt')
            #    torch.save(attention_mask, '/cluster/home/wenkai/LAVIS/output/attention_mask.pt')

            # self.get_eval = False
            #'''
            #num_txt = 15
            #return_num_txt = 10
            with torch.no_grad():
                outputs = self.mistral_model.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, min_length=min_length,
                                                  max_length=max_length, temperature=temperature, return_dict_in_generate=True,
                                                  output_scores=True,
                                                  repetition_penalty=repetition_penalty, num_beams=num_beams,
                                                  length_penalty=length_penalty, num_return_sequences=num_captions,
                                                  eos_token_id=self.eos_token_id)
            output_text = self.mistral_tokenizer.batch_decode(outputs['sequences'], skip_special_tokens=True)
            '''
            num_txt = 5
            return_num_txt = 1
            with torch.no_grad():
                outputs = self.mistral_model.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, min_length=1,
                                                  max_length=96,temperature=1.,return_dict_in_generate=True, output_scores=True,
                                                  repetition_penalty=1., num_beams=num_txt,
                                                  length_penalty=1, num_return_sequences=return_num_txt,eos_token_id=self.eos_token_id)
            output_text = self.mistral_tokenizer.batch_decode(outputs['sequences'], skip_special_tokens=True)
            '''
            probs = F.softmax(outputs['sequences_scores'])
            # print(output_text)
            output_text = [x.replace('\n', '').strip() for x in output_text]

            output_text_ = []
            for i in range(len(label)):
                # output_text_.append(';'.join(output_text[i*return_num_txt:(i+1)*return_num_txt]))
                output_text_.append(process_text(output_text[i * num_captions:(i + 1) * num_captions],
                                                 probs[i * num_captions:(i + 1) * num_captions]))
            #output_text_ = ['; '.join(list(set([i.strip() for i in x.split(';')]))) for x in output_text_]
            # with open('/cluster/home/wenkai/LAVIS/output/mf_bp_cc/output_test_mf_exp_493552.txt', 'a+',  encoding="utf-8") as f:
            #     for i in range(len(label)):
            #         f.write(name[i] + "|" +output_text_[i]+"|"+label[i]+'\n')
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

        inputs_mistral = self.mistral_proj(query_output.last_hidden_state)
        atts_mistral = torch.ones(inputs_mistral.size()[:-1], dtype=torch.long).to(self.device)

        label = samples["text_input"]
        name = samples['name']
        text = samples['prompt']
        # text = ['' for i in range(len(label))]
        mistral_tokens = self.mistral_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(self.device)
        # inputs_embeds = self.mistral_model.model.decoder.embed_tokens(mistral_tokens.input_ids)
        inputs_embeds = self.mistral_model.model.embed_tokens(mistral_tokens.input_ids)
        inputs_embeds = torch.cat([inputs_mistral, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_mistral, mistral_tokens.attention_mask], dim=1)
        # if name[0] == 'Pin':
        #    torch.save(inputs_embeds, '/cluster/home/wenkai/LAVIS/output/inputs_embeds.pt')
        #    torch.save(attention_mask, '/cluster/home/wenkai/LAVIS/output/attention_mask.pt')

        # self.get_eval = False
        # '''
        # num_txt = 15
        # return_num_txt = 10
        num_txt = 15
        return_num_txt = 10
        with torch.no_grad():
            outputs = self.mistral_model.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                                                  min_length=1,
                                                  max_length=32, temperature=1., return_dict_in_generate=True,
                                                  output_scores=True,
                                                  repetition_penalty=1., num_beams=num_txt,
                                                  length_penalty=0., num_return_sequences=return_num_txt,
                                                  eos_token_id=self.eos_token_id)
        output_text = self.mistral_tokenizer.batch_decode(outputs['sequences'], skip_special_tokens=True)
        '''
        num_txt = 5
        return_num_txt = 1
        with torch.no_grad():
            outputs = self.mistral_model.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, min_length=1,
                                              max_length=96,temperature=1.,return_dict_in_generate=True, output_scores=True,
                                              repetition_penalty=1., num_beams=num_txt,
                                              length_penalty=1, num_return_sequences=return_num_txt,eos_token_id=self.eos_token_id)
        output_text = self.mistral_tokenizer.batch_decode(outputs['sequences'], skip_special_tokens=True)
        '''
        probs = F.softmax(outputs['sequences_scores'])
        # print(output_text)
        output_text = [x.replace('\n', '').strip() for x in output_text]

        output_text_ = []
        for i in range(len(label)):
            # output_text_.append(';'.join(output_text[i*return_num_txt:(i+1)*return_num_txt]))
            output_text_.append(process_text(output_text[i * return_num_txt:(i + 1) * return_num_txt],
                                             probs[i * return_num_txt:(i + 1) * return_num_txt]))
        return output_text_

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

        get_eval = cfg.get("get_eval", False)
        esm_size = cfg.get("esm_size", '650m')
        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 128)
        max_protein_len = cfg.get("max_protein_len", 128)

        apply_lemmatizer = cfg.get("apply_lemmatizer", False)

        model = cls(
            num_query_token=num_query_token,
            prompt=prompt,
            max_txt_len=max_txt_len,
            max_protein_len=max_protein_len,
            apply_lemmatizer=apply_lemmatizer,
            get_eval=get_eval,
            esm_size=esm_size,
        )
        model.load_checkpoint_from_config(cfg)

        return model
