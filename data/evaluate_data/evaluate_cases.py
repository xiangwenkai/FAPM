import pandas as pd
import re
import random
import Levenshtein
import numpy as np
import difflib
# from torchmetrics.text import BLEUScore
import time
from multiprocessing import Pool, Queue, Process
import matplotlib.pyplot as plt
from data.evaluate_data.utils import Ontology
# bleu = BLEUScore(n_gram=1)

def fuzzy_match(texts):
    text_dict = {}
    for context in texts:
        if context not in choices:
            # txt_dict[txt] = process.extractOne(txt, choices)[0]
            text_dict[context] = difflib.get_close_matches(context, choices, n=1, cutoff=0.)[0]
    return text_dict


def get_sim(text, label):
    all_s = []
    for x in label:
        s = 0
        for y in text:
            temp = Levenshtein.ratio(x, y)
            if temp > s:
                s = temp
        all_s.append(s)
    all_s = [round(i, 3) for i in all_s]

    # bs = [bleu(x, [label]) for x in text]
    return all_s


def txt_map(x, txt_dict):
    if type(x) == str:
        x = eval(x)
    x_ = []
    for i in x:
        if i == '':
            continue
        if i in txt_dict:
            x_.append(txt_dict[i])
        else:
            x_.append(i)
    return x_


def go_map(t):
    if t in GO_dict:
        return GO_dict[t]
    else:
        print(t)


def get_term(df):
    from collections import Counter
    cnt = Counter()
    for i, row in enumerate(df.itertuples()):
        for term in row.prop_annotations:
            cnt[term] += 1
    terms = list(cnt.keys())
    # remove top
    for top_term in ['GO:0005575', 'GO:0003674', 'GO:0008150']:
        if top_term in terms:
            terms.remove(top_term)
    terms_df = pd.DataFrame({'gos': terms})
    terms_df.to_pickle(f'/cluster/home/wenkai/deepgozero/data/blip2/terms.pkl')


if __name__ == "__main__":
    go = Ontology(f'/cluster/home/wenkai/deepgozero/data/data/go.obo', with_rels=True)
    go_des = pd.read_csv('/cluster/home/wenkai/LAVIS/data/go_descriptions_new.txt', sep='|', header=None)
    go_des.columns = ['GO', 'function']
    go_des = go_des[go_des['function'].notnull()]
    go_des['function'] = go_des['function'].apply(lambda x: x.lower().strip())
    go_des['GO'] = go_des['GO'].apply(lambda x: re.sub('_', ':', x))
    GO_dict = dict(zip(go_des['function'], go_des['GO']))

    data = pd.read_csv('/cluster/home/wenkai/LAVIS/output/output_case.txt', sep='|', header=None)
    data.columns = ['protein', 'pred', 'label']
    data['label'] = data['label'].apply(lambda x: x.lower())
    data['pred'] = data['pred'].apply(lambda x: re.sub('</s>', '', x))

    data['label_list'] = data['label'].apply(lambda x: [i.strip() for i in x.split(';')])
    data['pred_list'] = data['pred'].apply(lambda x: [i.strip() for i in x.split(';')])

    test = pd.read_csv('/cluster/home/wenkai/LAVIS/data/pretrain/test.csv', sep='|')
    test = test.drop_duplicates()
    test['function'] = test['function'].apply(lambda x: x.lower().strip())
    test['function'] = test['function'].apply(lambda x: [i.strip() for i in x.split(';')])
    test['GO_label'] = test['GO_label'].apply(lambda x: [i.strip() for i in x.split(';')])

    test_dict = dict()
    for x, y in zip(test['function'], test['GO_label']):
        temp = dict(zip(x, y))
        test_dict.update(temp)
    GO_dict.update(test_dict)

    choices = list(test_dict.keys())

    ### 预测的文本如果不在GO标签词中，则算作最相似的GO标签
    '''
    print("找到与预测文本最相似的GO标签......")
    t0 = time.time()
    txt_dict = {}

    all_txt = []
    for txt in data['pred_list']:
        if type(txt) == str:
            all_txt.extend(eval(txt))
        else:
            all_txt.extend(txt)
    all_txt = list(set(all_txt))

    n = len(all_txt)
    thread = 10
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

    data['pred_list'] = data['pred_list'].apply(lambda x: txt_map(x, txt_dict))
    data['pred_list'] = data['pred_list'].apply(lambda x: list(set(x)))
    print("fuzzy matching time: {}".format(time.time() - t0))

    print("calculating f1 score ......")
    data['label_list_go'] = data['label_list'].apply(lambda x: [go_map(i) for i in x])
    data['pred_list_go'] = data['pred_list'].apply(lambda x: [go_map(i) for i in x])
    '''

    # 准备case测试数据：blip2预测的Go标签作为feature，label加入祖先后作为预测的Y
    prepare_ancestors = True
    if prepare_ancestors:
        print("准备加入祖先后的数据......")
        def prop(df):
            prop_annotations = []
            for i, row in df.iterrows():
                # Propagate annotations
                annot_set = set()
                annots = row['GO_label']
                for go_id in annots:
                    annot_set |= go.get_anchestors(go_id)
                annots = list(annot_set)
                prop_annotations.append(annots)
            df['prop_annotations'] = prop_annotations
            return df

        def pred_text_to_go(df):
            df['pred'] = df['pred'].apply(lambda x: re.sub('</s>', '', x))

            df['pred_list'] = df['pred'].apply(lambda x: [i.strip() for i in x.split(';')])
            ### 预测的文本如果不在GO标签词中，则算作最相似的GO标签
            t0 = time.time()
            txt_dict = {}

            all_txt = []
            for txt in df['pred_list']:
                if type(txt) == str:
                    all_txt.extend(eval(txt))
                else:
                    all_txt.extend(txt)

            all_txt = list(set(all_txt))
            if '' in all_txt:
                all_txt.remove('')

            n = len(all_txt)
            thread = 10
            size = int(n / thread)
            inds = list(range(0, n, size))
            inds.append(n)
            all_txt_sep = [all_txt[i: min(i + size, n)] for i in inds[:-1]]

            with Pool(processes=thread) as pool:
                result = pool.map(fuzzy_match, all_txt_sep)
            pool.close()
            pool.join()
            for d in result:
                txt_dict.update(d)

            # for txt in all_txt[:10]:
            #     fuzzy_match(txt)

            df['pred_list'] = df['pred_list'].apply(lambda x: txt_map(x, txt_dict))
            df['pred_list'] = df['pred_list'].apply(lambda x: list(set(x)))
            print("fuzzy matching time: {}".format(time.time() - t0))

            df['pred_list_go'] = df['pred_list'].apply(lambda x: [go_map(i) for i in x])
            return df


        test_pred = pd.read_csv('/cluster/home/wenkai/LAVIS/output/output_case.txt', sep='|', header=None)
        test_pred.columns = ['protein', 'pred', 'GO_label']
        test_pred['GO_label'] = test_pred['GO_label'].apply(lambda x: [i.strip() for i in x.split(';')])
        test_pred = prop(test_pred)
        test_pred = pred_text_to_go(test_pred)

        for cat in ['mf', 'bp', 'cc']:
            test_pred.to_pickle('/cluster/home/wenkai/deepgozero/data/blip2/{}/test_case.pkl'.format(cat))
