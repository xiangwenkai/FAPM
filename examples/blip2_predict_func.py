import os
import re

import torch
from PIL import Image

from lavis.models import load_model_and_preprocess
from lavis.processors import load_processor
from lavis.common.registry import registry
from torch.nn import functional as F
from lavis.models.base_model import all_gather_with_grad, concat_all_gather
import numpy as np
import pandas as pd
import time
from fuzzywuzzy import process
from multiprocessing import Pool, Queue, Process
import difflib
import Levenshtein
# import obonet


# setup device to use
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
# device = 'cpu'


def txt_map(x, txt_dict):
    if type(x) == str:
        x = eval(x)
    x_ = []
    for i in x:
        if i in txt_dict:
            x_.append(txt_dict[i])
        else:
            x_.append(i)
    return x_


def levenshtein_sim(text, label):
    all_s = []
    for x in label:
        s = 0
        for y in text:
            temp = Levenshtein.ratio(x, y)
            if temp > s:
                s = temp
        all_s.append(s)
    all_s = [round(i, 3) for i in all_s]
    return all_s

def func(text, label):
    all_s = []
    for x in text:
        s = 0
        for y in label:
            temp = Levenshtein.ratio(x, y)
            if temp > s:
                s = temp
        all_s.append(s)
    all_s = [round(i, 3) for i in all_s]
    return all_s


def stage2_output(df_test, return_num_txt=1):
    config = {'arch': 'blip2_protein_opt', 'load_finetuned': False,
              'pretrained': '/cluster/home/wenkai/LAVIS/lavis/output/BLIP2/Pretrain_stage2/20230924220/checkpoint_5.pth',
              'finetuned': '', 'num_query_token': 32, 'opt_model': 'facebook/opt-2.7b', 'prompt': '',
              'model_type': 'pretrain_protein_opt2.7b', 'load_pretrained': True, 'freeze_vit': True,
              'max_protein_len': 600,
              'max_txt_len': 25}

    model_cls = registry.get_model_class(config['arch'])
    model = model_cls.from_config(config)
    model.to(device)
    model.eval()

    images = df_test['protein'].tolist()
    n = len(images)
    bsz = 12
    iter = n // bsz + 1

    for i in range(iter):
        image = images[i*bsz: min(n, (i+1)*bsz)]
        image = [('protein{}'.format(i), x) for i, x in enumerate(image)]

        with model.maybe_autocast():
            _, _, batch_tokens = model.visual_encoder(image)
            image_embeds = model.ln_vision(batch_tokens.to(device), repr_layers=[model.vis_layers], return_contacts=True)["representations"][model.vis_layers].contiguous()

        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

        query_tokens = model.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = model.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        inputs_opt = model.opt_proj(query_output.last_hidden_state)
        atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(device)

        model.opt_tokenizer.padding_side = "right"

        text = ['' for i in range(len(image))]
        opt_tokens = model.opt_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=model.max_txt_len,
        ).to(device)
        inputs_embeds = model.opt_model.model.decoder.embed_tokens(opt_tokens.input_ids)
        inputs_embeds = torch.cat([inputs_opt, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)
        num_txt = 6
        with model.maybe_autocast():
            outputs = model.opt_model.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, min_length=3,
                                               max_length=30,
                                               repetition_penalty=1., num_beams=num_txt, eos_token_id=50118,
                                               length_penalty=1., num_return_sequences=return_num_txt, temperature=1.)
        output_text = model.opt_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        output_text = [text.strip() for text in output_text]
        output_text_ = []
        for i in range(len(image)):
            output_text_.append(';'.join(output_text[i * return_num_txt:(i + 1) * return_num_txt]))
        with open('/cluster/home/wenkai/LAVIS/output/output{}.txt'.format(fix), 'a+') as f:
            for i in range(len(image)):
                f.write(image[i][1] + "|" + output_text_[i] + '\n')


cat = 'mf'
fix = '_mf'
if cat == 'bp':
    fix = '_bp'
if cat == 'cc':
    fix = '_cc'

return_num_txt = 1
# graph = obonet.read_obo("http://purl.obolibrary.org/obo/go.obo")

### Levenshtein similarity
test = pd.read_csv('/cluster/home/wenkai/LAVIS/data/sim_split/test{}.csv'.format(fix), sep='|')
test['function'] = test['function'].apply(lambda x: x.lower())


if os.path.exists('/cluster/home/wenkai/LAVIS/output/output{}.txt'.format(fix)):
    os.remove('/cluster/home/wenkai/LAVIS/output/output{}.txt'.format(fix))
print("stage 2 predict starting")
stage2_output(test)
print("stage 2 predict completed")

df_pred = pd.read_csv('/cluster/home/wenkai/LAVIS/output/output{}.txt'.format(fix), sep='|', header=None, on_bad_lines='warn')
df_pred.columns = ['protein', 'function']
df_pred = df_pred.drop_duplicates()
df_pred['function'] = df_pred['function'].apply(lambda x: str(x).split(';'))
df_pred['function'] = df_pred['function'].apply(lambda x: [i.strip() for i in list(set(x))])

test_g = test.groupby(['protein']).agg({'function': lambda x: list(x)}).reset_index()
test_g.columns = ['protein', 'label']

data = pd.merge(df_pred, test_g, on='protein', how='left')
data = data[data['label'].notnull()]

sim = []
for text, label in zip(data['function'].tolist(), data['label'].tolist()):
    sim.append(func(text, label))

data['sim'] = sim
data['avg_score'] = data['sim'].apply(lambda x: round(np.mean(x), 3))
data['count'] = data['sim'].apply(lambda x: x.count(1.))
print("average similarity score: {}".format(round(data['avg_score'].mean(), 3)))
print("Return texts: {}; Accuracy: {}".format(return_num_txt, data['count'].sum()/(return_num_txt*data.shape[0])))
data.to_csv('/cluster/home/wenkai/LAVIS/output/predict_{}.csv'.format(cat), index=False, sep='|')




