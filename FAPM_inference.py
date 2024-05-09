import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
from lavis.models.protein_models.protein_function_opt import Blip2ProteinMistral
# from lavis.models.blip2_models.blip2_opt import Blip2ProteinOPT
from data.evaluate_data.utils import Ontology
import random
import difflib
import re


def process_text(txts, probs):
    res = dict()
    for txt, prob in zip(txts, probs):
        txt_sep = [x.strip() for x in txt.split(';')]
        for txt_sub in txt_sep:
            txt_sub = txt_sub.replace('|', '')
            if txt_sub not in res and txt_sub != '':
                res[txt_sub] = round(prob.item(),3)
    return '; '.join([str((k, v)) for k, v in res.items()])

# model = Blip2ProteinOPT(esm_size='3b')
model = Blip2ProteinMistral(esm_size='3b')
model.load_checkpoint('model_save/checkpoint_mf2.pth')
# model.load_checkpoint('/cluster/home/wenkai/LAVIS/lavis/output/BLIP2/Pretrain_stage2/20240327081/checkpoint_2.pth')

esm_emb = torch.load('/cluster/home/wenkai/LAVIS/data/pretrain/ipr_domain_emb_esm2_3b/P18281.pt')['representations'][36]
esm_emb = F.pad(esm_emb.t(), (0, 1024 - len(esm_emb))).t()
samples = {'name': 'P18281',
           'image': torch.unsqueeze(esm_emb, dim=0),
           'text_input': 'actin monomer binding',
           'prompt': 'Acanthamoeba'}

prediction = model.generate(samples)

godb = Ontology(f'/cluster/home/wenkai/LAVIS/data/go1.4-basic.obo', with_rels=True)

go_des = pd.read_csv('data/go_descriptions1.4.txt', sep='|', header=None)
go_des.columns = ['id', 'text']
go_des = go_des.dropna()
go_des['id'] = go_des['id'].apply(lambda x: re.sub('_', ':', x))
go_obo_set = set(go_des['id'].tolist())
go_des['text'] = go_des['text'].apply(lambda x: x.lower())
GO_dict = dict(zip(go_des['text'], go_des['id']))

# terms_mf = pd.read_pickle('/cluster/home/wenkai/deepgo2/data/mf/terms.pkl')
terms_mf = pd.read_pickle('data/terms/terms.pkl')
choices_mf = list(set(terms_mf['gos']))
choices = {x.lower(): x for x in choices_mf}

pred_terms_list = []
pred_go_list = []
prop_annotations = []
for x in prediction:
    pred_terms = []
    pred_go = []
    annot_set = set()
    for i in x:
        txt = i[0]
        prob = i[1]
        sim_list = difflib.get_close_matches(txt.lower(), choices, n=1, cutoff=0.9)
        if len(sim_list) > 0:
            pred_terms.append((sim_list[0], prob))
            pred_go.append((GO_dict[sim_list[0]], prob))
            annot_set |= godb.get_anchestors(GO_dict[sim_list[0]])
    pred_terms_list.append(pred_terms)
    pred_go_list.append(pred_go)
    annots = list(annot_set)
    prop_annotations.append(annots)



