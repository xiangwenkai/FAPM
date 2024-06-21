import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
from lavis.models.protein_models.protein_function_opt import Blip2ProteinMistral
# from lavis.models.base_model import FAPMConfig
# from lavis.models.blip2_models.blip2_opt import Blip2ProteinOPT
import random
from lavis.models.base_model import FAPMConfig
import argparse

prop = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FAPM')
    parser.add_argument('--model_path', type=str, help='Dataset path')
    parser.add_argument('--example_path', type=str, help='Example protein path')
    parser.add_argument('--device', type=str, default='cuda', help='Which gpu to use if any (default: cuda)')
    parser.add_argument('--prompt', type=str, default='none', help='Input prompt for protein function prediction')
    parser.add_argument('--ground_truth', type=str, default='none', help='ground truth function')
    args = parser.parse_args()
    test_sdf_paths = args.model_path

    # model = Blip2ProteinOPT(config=FAPMConfig(), esm_size='3b')
    # model.load_checkpoint('/cluster/home/wenkai/LAVIS/lavis/output/BLIP2/Pretrain_stage2/20240327081/checkpoint_2.pth')
    model = Blip2ProteinMistral(config=FAPMConfig(), esm_size='3b')
    model.load_checkpoint(args.model_path)
    model.to(args.device)

    # esm_emb = torch.load('/cluster/home/wenkai/LAVIS/data/pretrain/ipr_domain_emb_esm2_3b/Gp49.pt')['representations'][36]
    esm_emb = torch.load(args.example_path)['representations'][36]
    esm_emb = F.pad(esm_emb.t(), (0, 1024 - len(esm_emb))).t().to('cuda')
    samples = {'name': ['P18281'],
               'image': torch.unsqueeze(esm_emb, dim=0),
               'text_input': [args.ground_truth],
               'prompt': [args.prompt]}
    prediction = model.generate(samples, length_penalty=0., num_beams=15, num_captions=10, temperature=1., repetition_penalty=1.0)
    print(f"Text Prediction: {prediction}")


    if prop == True:
        from data.evaluate_data.utils import Ontology
        import difflib
        import re

        # godb = Ontology(f'/cluster/home/wenkai/LAVIS/data/go1.4-basic.obo', with_rels=True)
        godb = Ontology(f'data/go1.4-basic.obo', with_rels=True)

        go_des = pd.read_csv('data/go_descriptions1.4.txt', sep='|', header=None)
        go_des.columns = ['id', 'text']
        go_des = go_des.dropna()
        go_des['id'] = go_des['id'].apply(lambda x: re.sub('_', ':', x))
        go_obo_set = set(go_des['id'].tolist())
        go_des['text'] = go_des['text'].apply(lambda x: x.lower())
        GO_dict = dict(zip(go_des['text'], go_des['id']))
        Func_dict = dict(zip(go_des['id'], go_des['text']))

        # terms_mf = pd.read_pickle('/cluster/home/wenkai/deepgo2/data/mf/terms.pkl')
        terms_mf = pd.read_pickle('data/terms/mf_terms.pkl')
        choices_mf = [Func_dict[i] for i in list(set(terms_mf['gos']))]
        choices = {x.lower(): x for x in choices_mf}

        pred_terms_list = []
        pred_go_list = []
        prop_annotations = []
        for x in prediction:
            x = [eval(i) for i in x.split('; ')]
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

        print(f"Predictions of GO terms: \n{pred_terms_list} \nPredictions of GO id: \n{pred_go_list} \nPredictions of GO id propgated: \n{prop_annotations}")


