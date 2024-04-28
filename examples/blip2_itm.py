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
import os
# import obonet


def fuzzy_match(texts):
    text_dict = {}
    for context in texts:
        if context not in choices:
            # txt_dict[txt] = process.extractOne(txt, choices)[0]
            text_dict[context] = difflib.get_close_matches(context, choices, n=1, cutoff=0.)[0]
    return text_dict


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
    for x in label:
        s = 0
        for y in text:
            temp = Levenshtein.ratio(x, y)
            if temp > s:
                s = temp
        all_s.append(s)
    all_s = [round(i, 3) for i in all_s]
    return all_s


def stage2_output(df_test):
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
        num_txt = 10
        return_num_txt = 5
        with model.maybe_autocast():
            outputs = model.opt_model.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, min_length=3,
                                               max_length=30,
                                               repetition_penalty=5., num_beams=num_txt, eos_token_id=50118,
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

# model_pth = {'mf': 'uniprot_swissprot_mf_stage1_epo19.pth', 'bp': 'checkpoint17_GO_swissprot_reviewed_bp_stage1.pth', 'cc': ''}

# graph = obonet.read_obo("http://purl.obolibrary.org/obo/go.obo")

# setup device to use
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
# device = 'cpu'

### Levenshtein similarity
test = pd.read_csv('/cluster/home/wenkai/LAVIS/data/sim_split/test{}.csv'.format(fix), sep='|')[:10000]
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

test.columns
test_g = test.groupby(['protein']).agg({'function': lambda x: list(x)}).reset_index()
test_g.columns = ['protein', 'label']

data = pd.merge(df_pred, test_g, on='protein', how='left')
data = data[data['label'].notnull()]

sim = []
for text, label in zip(data['function'].tolist(), data['label'].tolist()):
    sim.append(func(text, label))

data['sim'] = sim
data['avg_score'] = data['sim'].apply(lambda x: round(np.mean(x), 3))
print("average similarity score: {}".format(round(data['avg_score'].mean(), 3)))
# data.to_csv('/home/nilin/LAVIS/predict_{}.csv'.format(cat), index=False, sep='|')


test = pd.read_csv('/cluster/home/wenkai/LAVIS/data/sim_split/test{}.csv'.format(fix), sep='|', usecols=['function', 'GO_label'])
test['function'] = test['function'].apply(lambda x: x.lower())
test = test.drop_duplicates()
test_dict = dict(zip(test['function'], test['GO_label']))
val = pd.read_csv('/cluster/home/wenkai/LAVIS/data/sim_split/val{}.csv'.format(fix), sep='|', usecols=['function', 'GO_label'])
val['function'] = val['function'].apply(lambda x: x.lower())
val = val.drop_duplicates()
val_dict = dict(zip(val['function'], val['GO_label']))
train = pd.read_csv('/cluster/home/wenkai/LAVIS/data/sim_split/train{}.csv'.format(fix), sep='|', usecols=['function', 'GO_label'])
train['function'] = train['function'].apply(lambda x: x.lower())
train = train.drop_duplicates()
train_dict = dict(zip(train['function'], train['GO_label']))


# go_des = pd.read_csv('/home/nilin/LAVIS/data/go_descriptions_new.txt', sep='|', header=None)
# # go_des = pd.read_csv('/home/nilin/LAVIS/data/go_descriptions.txt', sep='|', header=None)
# go_des.columns = ['GO', 'function']
# go_des = go_des[go_des['function'].notnull()]
# go_des['function'] = go_des['function'].apply(lambda x: x.lower())
# GO_dict = dict(zip(go_des['function'], go_des['GO']))
GO_dict = {}
GO_dict.update(train_dict)
GO_dict.update(val_dict)
GO_dict.update(test_dict)
choices = list(GO_dict.keys())



# data = pd.read_csv('/home/nilin/LAVIS/predict_{}.csv'.format(cat), sep='|')
data = data.sort_values(by='protein')
data = data.drop_duplicates('protein')
# data = data.sample(1000)

### 预测的文本如果不在GO标签词中，则算作最相似的GO标签
t0 = time.time()
txt_dict = {}

all_txt = []
for txt in data['function']:
    if type(txt) == str:
        all_txt.extend(eval(txt))
    else:
        all_txt.extend(txt)
all_txt = list(set(all_txt))

n = len(all_txt)
thread = 20
size = int(n/thread)
inds = list(range(0, n, size))
inds.append(n)
all_txt_sep = [all_txt[i: min(i+size, n)] for i in inds[:-1]]

with Pool(processes=thread) as pool:
    result = pool.map(fuzzy_match, all_txt_sep)
pool.close()
pool.join()
for d in result:
    txt_dict.update(d)

# for txt in all_txt[:10]:
#     fuzzy_match(txt)

data['function'] = data['function'].apply(lambda x: txt_map(x, txt_dict))
data['function'] = data['function'].apply(lambda x: list(set(x)))
print("fuzzy matching time: {}".format(time.time() - t0))




### Find the generated GO text that not included in the ground truth. Then generate pairs between them.
# pair_a, pair_b = [], []
# for preds, labels in zip(data['function'], data['label']):
#     if type(preds) == str:
#         preds = eval(preds)
#     if type(labels) == str:
#         labels = eval(labels)
#     l = len(labels)
#     for pred in preds:
#         if pred not in labels:
#             pair_a.extend([pred]*l)
#             pair_b.extend(labels[:])
# pair_a = [re.sub('_', ':', GO_dict[i]) for i in pair_a]
# pair_b = [re.sub('_', ':', GO_dict[i]) for i in pair_b]
# with open('/home/nilin/LAVIS/examples/GO_pair{}.txt'.format(fix), 'w+') as f:
#     for i, j in zip(pair_a, pair_b):
#         f.write(i+' '+j+'\n')


# load model
model_config = {'arch': 'blip2_protein', 'load_finetuned': False,
                'pretrained': '/cluster/home/wenkai/LAVIS/lavis/output/BLIP2/Pretrain_stage1/20230922185/checkpoint_15.pth',
                'finetuned': '', 'num_query_token': 32, 'prompt': '',
                'model_type': 'pretrain', 'load_pretrained': True, 'freeze_vit': False,
                'max_protein_len': 512, 'max_txt_len': 25}

model_cls = registry.get_model_class(model_config['arch'])
model = model_cls.from_config(model_config)
model = model.to(device)
model.eval()

# evaluate
t0 = time.time()
proteins = list(data['protein'])
txts = list(data['function'])
scores = []
for seq, txt in zip(proteins, txts):
    image = [('protein1', seq)]
    _, _, batch_tokens = model.visual_encoder(image)
    image_embeds = model.ln_vision(batch_tokens.to(device), repr_layers=[30], return_contacts=True)["representations"][
        30].contiguous()

    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

    query_tokens = model.query_tokens.expand(image_embeds.shape[0], -1, -1)

    query_output = model.Qformer.bert(
        query_embeds=query_tokens,
        encoder_hidden_states=image_embeds,
        encoder_attention_mask=image_atts,
        use_cache=True,
        return_dict=True,
    )

    image_feats = F.normalize(model.vision_proj(query_output.last_hidden_state), dim=-1)

    image_feats_all = concat_all_gather(image_feats)

    if type(txt) == str:
        txt = eval(txt)
    length = len(txt)
    with torch.no_grad():
        text_tokens = model.tokenizer(
            txt,
            padding="max_length",
            truncation=True,
            max_length=model.max_txt_len,
            return_tensors="pt",
        ).to(device)
        text_output = model.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )

        text_feat = F.normalize(
            model.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        )

        text_feat_all = concat_all_gather(text_feat)
        sim_q2t = torch.matmul(image_feats.unsqueeze(1), text_feat_all.unsqueeze(-1)).squeeze()
        sim_i2t, _ = sim_q2t.max(-1)
        # print('sim_i2t: {}'.format(sim_i2t))
    if length > 1:
        scores.append(list(sim_i2t.detach().cpu().numpy()))
    else:
        scores.append([sim_i2t.item()])
print("model evaluate time: {}".format(time.time() - t0))
data['score'] = scores

# precision and recall top-k
topk = 2
threshould = 0.1
labels = []
pred_labels = []
for l in data['label']:
    if type(l) == str:
        l = eval(l)
    labels.extend(l)

labels = list(set(labels))
total = len(labels)
for topk in range(1,7):
    for threshould in range(1, 25, 1):
        threshould /= 100
        filter_txts = []
        recalls = []
        precisions = []
        f1 = []
        tp_dict, fp_dict, fn_dict = dict(zip(labels, [0]*len(labels))), dict(zip(labels, [0]*len(labels))), dict(zip(labels, [0]*len(labels)))
        for txts, scores, label in zip(data['function'], data['score'], data['label']):
            if type(label) == str:
                label = eval(label)
            txts_ = np.array(txts)
            scores = np.array(scores)
            txts = txts_[scores > threshould]
            if len(txts) < 1:
                txts = txts_[np.argmax(scores)]
            scores = scores[scores > threshould]
            
            l = len(scores)
            ll = len(label)
            if l <= topk:
                filter_txts.append(list(txts))
            else:
                ind = np.argpartition(scores, -topk)[-topk:]
                txts = txts[ind]
                filter_txts.append(list(txts))
                l = topk
            for t in label:
                if t in txts:
                    tp_dict[t] += 1
                else:
                    fn_dict[t] += 1
            for p in txts:
                if p not in label:
                    if p in fp_dict:
                        fp_dict[p] += 1
                    else:
                        fp_dict[p] = 1
            pred_labels.extend(txts)
        p_total = len(set(pred_labels))
        re, pr = 0., 0.
        for x in labels:
            re += tp_dict[x] / (1.0 * (tp_dict[x] + fn_dict[x] + 1e-8))
            pr += tp_dict[x] / (1.0 * (tp_dict[x] + fp_dict[x]+1e-8))
        r = re / total
        p = pr / total
        f1 = 2 * p * r / (p + r)
        print("Topk: {}, threshould: {}, macro_recall: {}, macro_precision: {}, micro_f1: {}".format(topk, threshould, r, p, f1))
        #     num_r = 0
        #     num_p = 0
        #     for x in label:
        #         if x in txts:
        #             num_r += 1
        #     for x in txts:
        #         if x in label:
        #             num_p += 1
        #     recall = num_r/ll
        #     precision = num_p/(l+0.0001)
        #     recalls.append(recall)
        #     precisions.append(precision)
        #     f1.append((2*recall*precision)/(recall+precision+0.0001))
        #
        # data['predict'] = filter_txts
        # data['precision'] = precisions
        # data['recall'] = recalls
        # data['f1'] = f1
        # print("Topk: {}, threshould: {}, macro_recall: {}, macro_precision: {}, micro_f1: {}".format(topk, threshould, round(data['recall'].mean(), 4), round(data['precision'].mean(), 4), round(data['f1'].mean(), 4)))
    





# sim = []
# for text, label in zip(data['predict'].tolist(), data['label'].tolist()):
#     sim.append(levenshtein_sim(text, label))
#
# data['sim_filter'] = sim
# data['avg_score'] = data['sim_filter'].apply(lambda x: round(np.mean(x), 3))


# data['function'] = data['function'].apply(lambda x: eval(re.sub(';', ',', str(x))))
# data['label'] = data['label'].apply(lambda x: eval(re.sub(';', ',', str(x))))
# data['sim'] = data['sim'].apply(lambda x: eval(re.sub(';', ',', str(x))))
#
# data['function'] = data['function'].apply(lambda x: re.sub(',', ';', str(x)))
# data['label'] = data['label'].apply(lambda x: re.sub(',', ';', str(x)))
# data['sim'] = data['sim'].apply(lambda x: re.sub(',', ';', str(x)))
# data['predict'] = data['predict'].apply(lambda x: re.sub(',', ';', str(x)))
# data['sim_filter'] = data['sim_filter'].apply(lambda x: re.sub(',', ';', str(x)))

data.to_csv('/cluster/home/wenkai/LAVIS/output/predict_sim{}.csv'.format(fix), sep='|', index=False)
# data = pd.read_csv('/cluster/home/wenkai/LAVIS/output/predict_sim{}.csv'.format(fix), sep='|')








#
# # example
# image = ['MIELKHVTFGYNKKQMVLQDINITIPDGENVGILGESGCGKSTLASLVLGLFKPVKGEIYLSDNAVLTIFQHPLTSFNPDWTIETSLKEALYYYRGLTDNTAQDQLLLQHLSTFELNAQLLTKLPSEVSGGQLQRFNVMRSLLAQPRVLICDEITSNLDVIAEQNVINILKAQTITNLNHFIVISHDLSVLQRLVNRIIVLKDGMIVDDFAIEELFNVDRHPYTKELVQTFSY']
# image = [('protein{}'.format(i), x) for i, x in enumerate(image)]
#
# _, _, batch_tokens = model.visual_encoder(image)
# image_embeds = model.ln_vision(batch_tokens.to(device), repr_layers=[30], return_contacts=True)["representations"][30].contiguous()
#
# image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)
#
# query_tokens = model.query_tokens.expand(image_embeds.shape[0], -1, -1)
#
# query_output = model.Qformer.bert(
#     query_embeds=query_tokens,
#     encoder_hidden_states=image_embeds,
#     encoder_attention_mask=image_atts,
#     use_cache=True,
#     return_dict=True,
# )
#
# image_feats = F.normalize(model.vision_proj(query_output.last_hidden_state), dim=-1)
#
# image_feats_all = concat_all_gather(image_feats)
#
# functions = ['transmembrane transporter activity', 'nickel cation transmembrane transporter activity', 'nickel cation binding', 'atp hydrolysis activity', 'atp hydrolysis', 'cadmium binding', 'abc-type nickel transmembrane transporter activity', 'abc-type nickel transporter activity', 'nickel transmembrane transporter activity', 'atp binding']
# for text in functions:
#     with torch.no_grad():
#         # text = 'flavin adenine dinucleotide binding'
#         text_tokens = model.tokenizer(
#             text,
#             padding="max_length",
#             truncation=True,
#             max_length=model.max_txt_len,
#             return_tensors="pt",
#         ).to(device)
#         text_output = model.Qformer.bert(
#             text_tokens.input_ids,
#             attention_mask=text_tokens.attention_mask,
#             return_dict=True,
#         )
#
#         text_feat = F.normalize(
#             model.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
#         )
#
#         text_feat_all = concat_all_gather(text_feat)
#         sim_q2t = torch.matmul(image_feats.unsqueeze(1), text_feat_all.unsqueeze(-1)).squeeze()
#         sim_i2t, _ = sim_q2t.max(-1)
#         print('sim_i2t: {}'.format(sim_i2t))
#
#         # # text-query similarity: [batch_size, batch_size*num_gpu, num_query_tokens]
#         # sim_t2q = torch.matmul(
#         #     text_feat.unsqueeze(1).unsqueeze(1), image_feats_all.permute(0, 2, 1)
#         # ).squeeze()
#         #
#         # # text-image similarity: aggregate across all query tokens
#         # sim_t2i, _ = sim_t2q.max(-1)
#         # print('sim_t2i: {}'.format(sim_t2i))



