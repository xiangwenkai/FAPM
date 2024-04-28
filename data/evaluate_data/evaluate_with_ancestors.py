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
    terms_df.to_pickle(f'/cluster/home/wenkai/deepgozero/data/blip2/{cat}/terms.pkl')


if __name__ == "__main__":
    cat = 'mf'

    go = Ontology(f'/cluster/home/wenkai/deepgozero/data/data/go.obo', with_rels=True)
    go_des = pd.read_csv('/cluster/home/wenkai/LAVIS/data/go_descriptions_new.txt', sep='|', header=None)
    go_des.columns = ['GO', 'function']
    go_des = go_des[go_des['function'].notnull()]
    go_des['function'] = go_des['function'].apply(lambda x: x.lower().strip())
    go_des['GO'] = go_des['GO'].apply(lambda x: re.sub('_', ':', x))
    GO_dict = dict(zip(go_des['function'], go_des['GO']))


    data = pd.read_csv('/cluster/home/wenkai/LAVIS/output/predict_concat_test{}.csv'.format(cat), sep='|')

    data['label'] = data['label'].apply(lambda x: x.lower())
    data['pred'] = data['pred'].apply(lambda x: re.sub('</s>', '', x))

    data['label_list'] = data['label'].apply(lambda x: [i.strip() for i in x.split(';')])
    data['pred_list'] = data['pred'].apply(lambda x: [i.strip() for i in x.split(';')])

    train = pd.read_csv('/cluster/home/wenkai/LAVIS/data/sim_split/train_{}.csv'.format(cat), sep='|')
    train = train.drop_duplicates()
    train['function'] = train['function'].apply(lambda x: x.lower().strip())
    train_dict = dict(zip(train['function'], train['GO_label']))
    test = pd.read_csv('/cluster/home/wenkai/LAVIS/data/sim_split/test_{}.csv'.format(cat), sep='|')
    test = test.drop_duplicates()
    test['function'] = test['function'].apply(lambda x: x.lower().strip())
    test_dict = dict(zip(test['function'], test['GO_label']))
    GO_dict.update(train_dict)
    GO_dict.update(test_dict)

    choices = []
    for x in data['label_list'].tolist() + train['function'].tolist():
        choices.extend(x)
    choices = list(set(choices))


    ### 预测的文本如果不在GO标签词中，则算作最相似的GO标签
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
    thread = 40
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


    # sims = []
    # for text, label in zip(data['pred_list'].tolist(), data['label_list'].tolist()):
    #     a = get_sim(text, label)
    #     sims.append(a)
    #
    # data['sim'] = sims
    # data['avg_sim'] = data['sim'].apply(lambda x: round(np.mean(x), 3))
    # print("simlarity: {}".format(data['avg_sim'].mean()))


    print("calculating f1 score ......")
    data['label_list_go'] = data['label_list'].apply(lambda x: [go_map(i) for i in x])
    data['pred_list_go'] = data['pred_list'].apply(lambda x: [go_map(i) for i in x])


    labels = []
    pred_labels = []
    for l in data['label_list_go']:
        if type(l) == str:
            l = eval(l)
        labels.extend(l)

    label_count = {}
    for x in labels:
        if x not in label_count:
            label_count[x] = 1
        else:
            label_count[x] += 1

    labels = list(set(labels))
    total = len(labels)
    recalls = []
    precisions = []
    tp_dict, fp_dict, fn_dict = dict(zip(labels, [0]*len(labels))), dict(zip(labels, [0]*len(labels))), dict(zip(labels, [0]*len(labels)))
    for preds, label in zip(data['pred_list_go'], data['label_list_go']):
        if type(label) == str:
            label = eval(label)
        if type(preds) == str:
            txts = eval(preds)
        ll = len(label)
        for t in label:
            supgo = go.get_anchestors(t)
            if supgo.intersection(set(preds)):
                tp_dict[t] += 1
            else:
                fn_dict[t] += 1
        for p in preds:
            supgo = go.get_anchestors(p)
            if not supgo.intersection(set(label)):
                if p in fp_dict:
                    fp_dict[p] += 1
                else:
                    fp_dict[p] = 1
        pred_labels.extend(preds)
    p_total = len(set(pred_labels))
    recall, pr = 0., 0.
    for x in labels:
        recall += tp_dict[x] / (1.0 * (tp_dict[x] + fn_dict[x] + 1e-8))
        pr += tp_dict[x] / (1.0 * (tp_dict[x] + fp_dict[x] + 1e-8))
    r = recall / total
    p = pr / p_total
    f1 = 2 * p * r / (p + r)

    print("preds not in labels: {}".format(len(list(fp_dict.keys())) - total))
    print("f1 score: {}".format(f1))

    '''
    cat_f1 = {}
    for x in labels:
        if tp_dict[x] + fn_dict[x] > 0:
            re = tp_dict[x] / (1.0 * (tp_dict[x] + fn_dict[x] + 1e-8))
            pr = tp_dict[x] / (1.0 * (tp_dict[x] + fp_dict[x] + 1e-8))
            cat_f1[x] = 2 * pr * re / (pr + re + 1e-10)
    
    plt.xlabel('f score')
    plt.ylabel('count')
    print(np.mean(list(cat_f1.values())))
    plt.hist(list(cat_f1.values()), color='red', bins=30)
    plt.show()
    
    xs, ys = [], []
    for x in labels:
        xs.append(label_count[x])
        ys.append(cat_f1[x])
    df_count = pd.DataFrame({'xs': xs, 'ys': ys})
    df_count['xs'].loc[df_count['xs'] > 10] = 11
    df_count['xs'] = df_count['xs'].astype(str)
    df_count1 = df_count.groupby('xs').mean().reset_index()
    df_count2 = df_count.groupby('xs').count().reset_index()
    
    plt.xlabel('label count')
    plt.ylabel('f score mean')
    df_count1['xs'] = df_count1['xs'].astype(int)
    plt.scatter(df_count1['xs'], df_count1['ys'], color='red')
    plt.show()
    
    plt.xlabel('label count')
    plt.ylabel('protein num')
    df_count2['xs'] = df_count2['xs'].astype(int)
    plt.bar(df_count2['xs'], df_count2['ys'], color='red')
    plt.show()
    '''


    # 准备数据：blip2预测的Go标签作为feature，label加入祖先后作为预测的Y
    print("准备加入祖先后的数据......")
    train = pd.read_csv('/cluster/home/wenkai/LAVIS/data/sim_split/train_{}.csv'.format(cat), sep='|')
    test = pd.read_csv('/cluster/home/wenkai/LAVIS/data/sim_split/test_{}.csv'.format(cat), sep='|')
    train = train.groupby('name').agg({'GO_label': list}).reset_index()
    test = test.groupby('name').agg({'GO_label': list}).reset_index()

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

    train = prop(train)
    test = prop(test)

    train_test = pd.concat([train, test])
    get_term(train_test)
    del train_test

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
        thread = 40
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


    train_pred = pd.read_csv('/cluster/home/wenkai/LAVIS/output/predict_concat_train{}.csv'.format(cat), sep='|')
    test_pred = pd.read_csv('/cluster/home/wenkai/LAVIS/output/predict_concat_test{}.csv'.format(cat), sep='|')

    train_pred = pred_text_to_go(train_pred)
    test_pred = pred_text_to_go(test_pred)

    train_data = pd.merge(train[['name', 'prop_annotations']],
                          train_pred[['name', 'pred_list_go']],
                          on='name', how='inner')
    train_data = train_data.drop_duplicates('name')
    train_data.to_pickle('/cluster/home/wenkai/deepgozero/data/blip2/{}/train_data.pkl'.format(cat))

    test_data = pd.merge(test[['name', 'prop_annotations']],
                         test_pred[['name', 'pred_list_go']],
                         on='name', how='inner')
    test_data = test_data.drop_duplicates('name')
    test_data.to_pickle('/cluster/home/wenkai/deepgozero/data/blip2/{}/test_data.pkl'.format(cat))
    test_data.to_pickle('/cluster/home/wenkai/deepgozero/data/blip2/{}/valid_data.pkl'.format(cat))

