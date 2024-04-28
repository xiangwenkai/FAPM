import re
import pandas as pd
import time
from multiprocessing import Pool
import difflib
from utils import Ontology
import os


def filter(x_list):
    new_go = []
    # x_list = [i.strip() for i in x.split(';')]
    for i in x_list:
        if i in filter_go:
            new_go.append(i)
    return '; '.join(new_go)


def fuzzy_match(texts):
    text_dict = {}
    for context in texts:
        if context in choices:
            text_dict[context] = context
        elif context not in choices:
            # txt_dict[txt] = process.extractOne(txt, choices)[0]
            sim_list = difflib.get_close_matches(context.lower(), choices, n=1, cutoff=0.9)
            if len(sim_list) > 0:
                text_dict[context] = sim_list[0]
            else:
                # text_dict[context] = ''
                pass
    return text_dict


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
            # x_.append(i)
            pass
    return x_


def go_map_prob(x, GO_dict):
    res = []
    for t in x:
        if t[0] in GO_dict:
            res.append((GO_dict[t[0]], t[1]))
        else:
            pass
            # print("{} not in GO_dict".format(t[0]))
    return res


def txt_map_prob(x, txt_dict):
    if type(x) == str:
        x = eval(x)
    x_ = []
    temp = set()
    for i in x:
        if i[0] == '':
            continue
        elif i[0] in txt_dict and txt_dict[i[0]] not in temp:
            x_.append((txt_dict[i[0]].lower(), i[1]))
            temp.add(txt_dict[i[0]])
        # elif i[0] not in txt_dict:
        #    x_.append((i[0].lower(), i[1]))
        #    temp.add(i[0])
        else:
            continue
    return x_


def go_map(x, GO_dict):
    res = []
    for t in x:
        if t in GO_dict:
            res.append(GO_dict[t])
        else:
            # pass
            print("{} not in GO_dict".format(t))
    return res


def prop(df):
    prop_annotations = []
    for i, row in df.iterrows():
        # Propagate annotations
        annot_set = set()
        annots = row['GO_label']
        for go_id in annots:
            annot_set |= godb.get_anchestors(go_id)
        annots = list(annot_set)
        prop_annotations.append(annots)
    df['prop_annotations'] = prop_annotations
    return df


def pred_text_to_go(df, with_prob=False):
    # df['pred'] = df['pred'].apply(lambda x: re.sub('</s>', '', x))
    if with_prob:
        df['pred_list_prob'] = df['pred'].apply(lambda x: [eval(i.strip()) for i in x.split(';')])
        df['pred_list'] = df['pred_list_prob'].apply(lambda x: [i[0] for i in x])
    else:
        df['pred_list'] = df['pred'].apply(lambda x: list(set([i.strip() for i in x.split(';')])))
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
    # print(txt_dict)
    # for txt in all_txt[:10]:
    #     fuzzy_match(txt)
    if with_prob:
        df['pred_list_prob'] = df['pred_list_prob'].apply(lambda x: txt_map_prob(x, txt_dict))
        print("fuzzy matching time: {}".format(time.time() - t0))
        df['pred_list_go_prob'] = df['pred_list_prob'].apply(lambda x: go_map_prob(x, GO_dict))
        n0 = df.shape[0]
        df['len'] = df['pred_list_go_prob'].apply(lambda x: len(x))
        df = df[df['len'] > 0]
        df = df.drop('len', axis=1)
        df = df.dropna()
        print('{}条数据，不为空的预测有{}条'.format(n0, df.shape[0]))
    else:
        df['pred_list'] = df['pred_list'].apply(lambda x: txt_map(x, txt_dict))
        df['pred_list'] = df['pred_list'].apply(lambda x: [i.lower() for i in list(set(x))])
        print("fuzzy matching time: {}".format(time.time() - t0))
        df['pred_list_go'] = df['pred_list'].apply(lambda x: go_map(x, GO_dict))

        n0 = df.shape[0]
        df['len'] = df['pred_list_go'].apply(lambda x: len(x))
        df = df[df['len'] > 0]
        df = df.drop('len', axis=1)
        df = df.dropna()
        print('{}条数据，不为空的预测有{}条'.format(n0, df.shape[0]))
    return df


def cal_f1(df):
    df['label_list_go'] = df['label'].apply(lambda x: [i.strip() for i in x.split(';')])
    df['pred_list_go'] = df['pred_list'].apply(lambda x: [i.strip() for i in x.split(';')])

    labels = []
    pred_labels = []
    for l in df['label_list_go']:
        labels.extend(l)

    label_count = {}
    for x in labels:
        if x not in label_count:
            label_count[x] = 1
        else:
            label_count[x] += 1

    labels = list(set(labels))
    total = len(labels)
    tp_dict, fp_dict, fn_dict = dict(zip(labels, [0] * len(labels))), dict(zip(labels, [0] * len(labels))), dict(
        zip(labels, [0] * len(labels)))
    for preds, label in zip(df['pred_list_go'], df['label_list_go']):
        for t in label:
            # supgo = godb.get_anchestors(t)
            # if supgo.intersection(set(preds)):
            if t in preds:
                tp_dict[t] += 1
            else:
                fn_dict[t] += 1
        for p in preds:
            # supgo = godb.get_anchestors(p)
            # if not supgo.intersection(set(label)):
            if p not in label:
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
    print("recall:{}; percision:{}; f1 score: {}".format(r, p, f1))


def cat_go(x):
    try:
        cat = godb.get_namespace(x)
    except:
        print("{} not found".format(x))
        return
    if cat == NAMESPACES['mf']:
        return 'mf'
    elif cat == NAMESPACES['bp']:
        return 'bp'
    elif cat == NAMESPACES['cc']:
        return 'cc'
    return


def remove_root(x):
    if 'molecular_function' in x:
        x.remove('molecular_function')
    if 'biological_process' in x:
        x.remove('biological_process')
    if 'cellular_component' in x:
        x.remove('cellular_component')
    return x

if __name__ == "__main__":
    NAMESPACES = {
        'cc': 'cellular_component',
        'mf': 'molecular_function',
        'bp': 'biological_process'
    }
    #if not os.path.exists('/cluster/home/wenkai/LAVIS/data/pretrain/mf_bp_cc/terms.pkl'):
    if 1==1:
        data = pd.read_csv('/cluster/home/wenkai/LAVIS/data/pretrain/swissprot_domain_and_train_exp_prompt_new.csv', sep='|')
        print('数据规模：{}'.format(data.shape[0]))
        # data['function'] = data['function'].apply(lambda x: re.sub('[FPC]:', '', x))
        # data.to_csv('swissprot_domain_and_train_exp.csv', sep='|', index=False)

        godb = Ontology(f'/cluster/home/wenkai/LAVIS/data/go1.4-basic.obo', with_rels=True)
        go_des = pd.read_csv('/cluster/home/wenkai/LAVIS/data/go_descriptions1.4.txt', sep='|', header=None)
        go_des.columns = ['id', 'text']
        go_des = go_des.dropna()
        go_des['id'] = go_des['id'].apply(lambda x: re.sub('_', ':', x))
        go_des['ont'] = go_des['id'].apply(lambda x: cat_go(x))
        go_des = go_des.dropna()
        go_obo_set = set(go_des['id'].tolist())
        go_des['text'] = go_des['text'].apply(lambda x: x.lower())

        data['GO_label'] = data['GO_label'].apply(lambda x: [i.strip() for i in x.split(';')])
        data = prop(data)

        # 加入父节点，得到完整的terms,映射表等等
        go_dict = {}
        for x_list in data['prop_annotations']:
            for goid in x_list:
                if goid in go_dict:
                    go_dict[goid] += 1
                else:
                    go_dict[goid] = 1
        df_stat = pd.DataFrame({'id': list(go_dict.keys()), 'count': list(go_dict.values())})
        data_gos = set(df_stat['id'].tolist())
        go_des = go_des[go_des['id'].isin(data_gos)]
        filter_go = data_gos.intersection(go_obo_set)
        print(f"包括父节点的GO有{len(data_gos)}个，其中在go1.4.obo中出现的GO有{len(filter_go)}个")

        go_des.to_pickle('/cluster/home/wenkai/LAVIS/data/pretrain/mf_bp_cc/go_des.pkl')
        id2text_dict = dict(zip(go_des['id'], go_des['text']))
        GO_dict = dict(zip(go_des['text'], go_des['id']))

        choices_mf = list(set(go_des[go_des['ont'] == 'mf']['text']))
        choices_bp = list(set(go_des[go_des['ont'] == 'bp']['text']))
        choices_cc = list(set(go_des[go_des['ont'] == 'cc']['text']))

        choices_mf = {x.lower(): x for x in choices_mf}
        choices_bp = {x.lower(): x for x in choices_bp}
        choices_cc = {x.lower(): x for x in choices_cc}

        data['GO_label'] = data['GO_label'].apply(lambda x: filter(x))
        data = data[data['GO_label'] != '']
        data['function'] = data['GO_label'].apply(lambda x: [id2text_dict[i.strip()] for i in x.split(';')])
        data['function'] = data['function'].apply(lambda x: '; '.join(x))

        terms = pd.DataFrame({'gos': list(filter_go)})
        terms.to_pickle('/cluster/home/wenkai/LAVIS/data/pretrain/mf_bp_cc/terms.pkl')
        terms.to_pickle('/cluster/home/wenkai/deepgozero/data/blip2/pretrain/terms.pkl')

        terms_mf = pd.DataFrame({'gos': list(set(go_des[go_des['ont'] == 'mf']['id']))})
        terms_mf.to_pickle('/cluster/home/wenkai/deepgozero/data/blip2/pretrain/mf/terms.pkl')
        terms_mf.to_pickle('/cluster/home/wenkai/deepgo2/data/mf/terms.pkl')
        terms_bp = pd.DataFrame({'gos': list(set(go_des[go_des['ont'] == 'bp']['id']))})
        terms_bp.to_pickle('/cluster/home/wenkai/deepgozero/data/blip2/pretrain/bp/terms.pkl')
        terms_bp.to_pickle('/cluster/home/wenkai/deepgo2/data/bp/terms.pkl')
        terms_cc = pd.DataFrame({'gos': list(set(go_des[go_des['ont'] == 'cc']['id']))})
        terms_cc.to_pickle('/cluster/home/wenkai/deepgozero/data/blip2/pretrain/cc/terms.pkl')
        terms_cc.to_pickle('/cluster/home/wenkai/deepgo2/data/cc/terms.pkl')
    else:
        godb = Ontology(f'/cluster/home/wenkai/LAVIS/data/go1.4-basic.obo', with_rels=True)
        terms = pd.read_pickle('/cluster/home/wenkai/LAVIS/data/pretrain/mf_bp_cc/terms.pkl')
        filter_go = set(terms['gos'].tolist())
 
        terms_mf = pd.read_pickle('/cluster/home/wenkai/deepgo2/data/mf/terms.pkl')
        terms_bp = pd.read_pickle('/cluster/home/wenkai/deepgo2/data/bp/terms.pkl')
        terms_cc = pd.read_pickle('/cluster/home/wenkai/deepgo2/data/cc/terms.pkl')

        choices_mf = {x.lower(): x for x in terms_mf['gos'].tolist()}
        choices_bp = {x.lower(): x for x in terms_bp['gos'].tolist()}
        choices_cc = {x.lower(): x for x in terms_cc['gos'].tolist()}

        go_des = pd.read_pickle('/cluster/home/wenkai/LAVIS/data/pretrain/mf_bp_cc/go_des.pkl')
        id2text_dict = dict(zip(go_des['id'], go_des['text']))
        GO_dict = dict(zip(go_des['text'], go_des['id']))

    # 对于预测文件，进行GO筛选，并用相似度算法匹配到filter_go；对于train test val 文件，进行GO筛选、加入祖先、加入interPro特征
    # 加入interpro特征
    df_interpro = pd.read_csv('/cluster/home/wenkai/LAVIS/data/uniprot_sprot_blip2_func_data.txt', sep='|',
                              nrows=546389,
                              header=None)
    df_interpro.columns = ['name', 'seq', 'go', 'text', 'evi', 'ipr']
    df_interpro = df_interpro[df_interpro['ipr'].notnull()]
    df_interpro['ipr'] = df_interpro['ipr'].apply(lambda x: [i.strip() for i in x.split(';')])

    iprs = []
    for x in df_interpro['ipr'].tolist():
        if len(x) > 0:
            iprs.extend(x)
    iprs = list(set(iprs))
    print("ipr个数：{}".format(len(iprs)))
    df_ipr = pd.DataFrame({'interpros': iprs})
    df_ipr.to_pickle('/cluster/home/wenkai/LAVIS/data/interpros.pkl')
    df_ipr.to_pickle('/cluster/home/wenkai/deepgozero/data/blip2/pretrain/interpros.pkl')


    '''
    # test cases
    df_real = pd.read_csv('/cluster/home/wenkai/LAVIS/data/pretrain/test_2000.csv', sep='|')
    df_real[col] = df_real[col].apply(lambda x: [i.strip() for i in x.split(';')])
    #df_real[col] = df_real[col].apply(lambda x: filter(x))
    df_real = df_real[df_real[col] != '']
    print(df_real.shape)
    #df_real['GO_label'] = df_real['GO_label'].apply(lambda x: [id2text_dict[i] for i in x])
    #df_real['GO_label'] = df_real['GO_label'].apply(lambda x: [GO_dict[i] for i in x])
    df_real = prop(df_real)
    #df_real['prop_annotations'] = df_real['prop_annotations'].apply(lambda x: [id2text_dict[i] for i in x])
    #df_real['prop_annotations'] = df_real['prop_annotations'].apply(lambda x: remove_root(x))
    #df_real['prop_annotations'] = df_real['prop_annotations'].apply(lambda x: list(set([GO_dict[i] for i in x])))
    for ont in ['mf', 'bp', 'cc']:
        file_name = 'output_{}_test_2000'.format(ont)
        if ont == 'mf':
            choices = choices_mf
        elif ont == 'bp':
            choices = choices_bp
        elif ont == 'cc':
            choices = choices_cc
        print("对{}预测文本进行标准化...".format(file_name))
        df_pred = pd.read_csv('/cluster/home/wenkai/LAVIS/output/{}.txt'.format(file_name), sep='|', header=None, on_bad_lines='skip')
        df_pred.columns = ['name', 'pred', 'label']
        n0 = df_pred.shape[0]
        df_pred = pred_text_to_go(df_pred, with_prob=True)
        print("{}中有{}条数据未能找到相似度高的GO描述".format(file_name, n0-df_pred.shape[0]))
        #df_pred['pred_list'] = df_pred['pred_list'].apply(lambda x: '; '.join(x))
        #cal_f1(df_pred)
        df_pred[['name', 'pred_list_prob', 'label']].to_csv('/cluster/home/wenkai/LAVIS/output/{}_standard.csv'.format(file_name), sep='|', index=False)
    
        df_pred = pd.merge(df_pred[['name', 'pred_list_go_prob']], df_interpro[['name', 'ipr']], on='name', how='left')
        df_pred['ipr'] = df_pred['ipr'].fillna("").apply(list)
        ipr_and_pred = []
        for x, y in zip(df_pred['ipr'], df_pred['pred_list_go_prob']):
            try:
                ipr_and_pred.append(x + y)
            except:
                ipr_and_pred.append(y)
        df_pred['ipr_and_pred'] = ipr_and_pred
        print(df_real.isnull().sum())
        df_pred = pd.merge(df_pred, df_real[['name', 'protein', 'prop_annotations']], on='name', how='left')
        #df_pred = df_pred.dropna()
        print(df_pred.shape)
        df_pred[['name', 'protein', 'ipr', 'pred_list_go_prob', 'ipr_and_pred', 'prop_annotations']].to_pickle(
                '/cluster/home/wenkai/deepgozero/data/blip2/pretrain/{}/test_2000_data.pkl'.format(ont))
    '''

    '''
    df_real = pd.read_csv('/cluster/home/wenkai/LAVIS/data/pretrain/nextprot_mf.csv', sep='|')
    df_real['GO_label'] = df_real['GO_label'].apply(lambda x: [i.strip() for i in x.split(';')])
    df_real['GO_label'] = df_real['GO_label'].apply(lambda x: [id2text_dict[i] for i in x])
    df_real['GO_label'] = df_real['GO_label'].apply(lambda x: [GO_dict[i] for i in x])
    df_real = prop(df_real)
    df_real['prop_annotations'] = df_real['prop_annotations'].apply(lambda x: [id2text_dict[i] for i in x])
    df_real['prop_annotations'] = df_real['prop_annotations'].apply(lambda x: remove_root(x))
    df_real['prop_annotations'] = df_real['prop_annotations'].apply(lambda x: list(set([GO_dict[i] for i in x])))
    
    file = 'output_nextprot'
    choices = choices_mf
    df_pred = pd.read_csv('/cluster/home/wenkai/LAVIS/output/{}.txt'.format(file), sep='|', header=None, on_bad_lines='skip')
    df_pred.columns = ['name', 'pred', 'label']
    df_pred = pred_text_to_go(df_pred, with_prob=True)
    df_pred[['name', 'pred_list_prob', 'label']].to_csv('/cluster/home/wenkai/LAVIS/output/{}_standard.csv'.format(file), sep='|', index=False)
    
    df_pred = pd.merge(df_pred, df_real[['name', 'protein', 'prop_annotations']], on='name', how='left')
    df_pred['ipr'] = [[] for _ in range(df_pred.shape[0])]
    df_pred['ipr_and_pred'] = df_pred['pred_list_go_prob']
    df_pred[['name', 'protein', 'ipr', 'pred_list_go_prob', 'ipr_and_pred', 'prop_annotations']].to_pickle(
                '/cluster/home/wenkai/deepgozero/data/blip2/pretrain/mf/nextprot_data.pkl')
    '''
    # '''
    cat_id = {'mf': '445772', 'bp': '496359', 'cc': '505955'}
    col = 'GO_label'
    for ont in ['mf', 'bp', 'cc']:
    #for ont in ['mf']:
        if ont == 'mf':
            choices = choices_mf
        elif ont == 'bp':
            choices = choices_bp
        elif ont == 'cc':
            choices = choices_cc
        for split in ['train', 'val', 'test']:
        #for split in ['test']:
            df_real = pd.read_csv(f'/cluster/home/wenkai/LAVIS/data/pretrain/mf_bp_cc/{split}_exp_{ont}_new.csv',
                                  sep='|')
            df_real[col] = df_real[col].apply(lambda x: [i.strip() for i in x.split(';')])
            df_real[col] = df_real[col].apply(lambda x: filter(x))
            df_real = df_real[df_real[col] != '']
            print(df_real.shape)
            df_real['GO_label'] = df_real['GO_label'].apply(lambda x: [i.strip() for i in x.split(';')])
            df_real['GO_label'] = df_real['GO_label'].apply(lambda x: [id2text_dict[i] for i in x])
            df_real['GO_label'] = df_real['GO_label'].apply(lambda x: [GO_dict[i] for i in x])
            df_real = prop(df_real)
            df_real['prop_annotations'] = df_real['prop_annotations'].apply(lambda x: [id2text_dict[i] for i in x])
            df_real['prop_annotations'] = df_real['prop_annotations'].apply(lambda x: remove_root(x))
            df_real['prop_annotations'] = df_real['prop_annotations'].apply(lambda x: list(set([GO_dict[i] for i in x])))

            # 预测text转为go
            df_pred = pd.read_csv(
                f'/cluster/home/wenkai/LAVIS/output/mf_bp_cc/output_{split}_{ont}_exp_{cat_id[ont]}.txt', sep='|',
                header=None, on_bad_lines='skip')
            df_pred.columns = ['name', 'pred', 'label']
            n0 = df_pred.shape[0]
            df_pred = pred_text_to_go(df_pred, with_prob=True)
            print("{}中有{}条数据未能找到相似度高的GO描述".format(ont, n0 - df_pred.shape[0]))
            df_pred[['name', 'pred_list_prob', 'label']].to_csv(
                f'/cluster/home/wenkai/LAVIS/output/mf_bp_cc/output_{split}_{ont}_{cat_id[ont]}_standard.csv', sep='|',
                index=False)

            df_pred = pd.merge(df_pred[['name', 'pred_list_go_prob']], df_interpro[['name', 'ipr']], on='name', how='left')
            df_pred['ipr'] = df_pred['ipr'].fillna("").apply(list)
            ipr_and_pred = []
            for x, y in zip(df_pred['ipr'], df_pred['pred_list_go_prob']):
                try:
                    ipr_and_pred.append(x + y)
                except:
                    ipr_and_pred.append(y)
            df_pred['ipr_and_pred'] = ipr_and_pred

            df_pred = pd.merge(df_pred, df_real[['name', 'protein', 'prop_annotations']], on='name', how='left')
            df_pred = df_pred.dropna()
            df_pred[['name', 'protein', 'ipr', 'pred_list_go_prob', 'ipr_and_pred', 'prop_annotations']].to_pickle(
                f'/cluster/home/wenkai/deepgozero/data/blip2/pretrain/{ont}/{split}_data_{cat_id[ont]}.pkl')
            df_pred[['name', 'protein', 'ipr', 'pred_list_go_prob', 'ipr_and_pred', 'prop_annotations']].to_pickle(
                f'/cluster/home/wenkai/deepgo2/data/{ont}/{split}_data_{cat_id[ont]}.pkl')
            if split == 'val':
                df_pred[['name', 'protein', 'ipr', 'pred_list_go_prob', 'ipr_and_pred', 'prop_annotations']].to_pickle(
                    f'/cluster/home/wenkai/deepgozero/data/blip2/pretrain/{ont}/valid_data_{cat_id[ont]}.pkl')
                df_pred[['name', 'protein', 'ipr', 'pred_list_go_prob', 'ipr_and_pred', 'prop_annotations']].to_pickle(
                    f'/cluster/home/wenkai/deepgo2/data/{ont}/valid_data_{cat_id[ont]}.pkl')
            print(f"{ont} {split} deepgozero propagation data completed")
    # '''
