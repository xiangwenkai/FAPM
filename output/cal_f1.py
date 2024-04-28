import pandas as pd


def cal_f1(df, standard=False):
    df['label_list'] = df['label'].apply(lambda x: [i.strip().lower() for i in x.split(';')])
    #df['pred_list_go'] = df['pred'].apply(lambda x: [i.strip() for i in x.split(';')])
    if standard:
        df['pred_list'] = df['pred'].apply(lambda x: [i[0] for i in eval(str(x))])
    else:
       df['pred_list_prob'] = df['pred'].apply(lambda x: [eval(i.strip()) for i in str(x).split(';')])
       df['pred_list'] = df['pred_list_prob'].apply(lambda x: [i[0] for i in x])

    labels = []
    pred_labels = []
    for l in df['label_list']:
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
    for preds, label in zip(df['pred_list'], df['label_list']):
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
    f1 = 2 * p * r / (p + r + 1e-8)

    print("preds not in labels: {}".format(len(list(fp_dict.keys())) - total))
    print("recall:{}; percision:{}; f1 score: {}".format(r, p, f1))


names = ['output_test_mf_exp_493552.txt', 'output_test_mf_exp_445772_pre.txt', 'output_test_mf_exp_445772.txt', 'output_test_mf_exp_486524.txt', 'output_test_mf_493552_standard.csv', 'output_test_mf_445772_standard.csv', 'output_test_mf_exp_445772_withprompt.txt', 'output_test_mf_exp_506753.txt']
#names = ['output_test_bp_exp_451674.txt', 'output_test_bp_exp_493547_pre.txt', 'output_test_bp_exp_496359_withprompt.txt']

for name in names:
    print(name)
    df = pd.read_csv('/cluster/home/wenkai/LAVIS/output/mf_bp_cc/{}'.format(name), sep='|', header=None)
    if df.iloc[0, 0] == 'name':
        df = df[1:]
    #print(df.shape)
    df.columns = ['name', 'pred', 'label']
    if 'standard' in name:
        cal_f1(df, standard=True)
    else:
        cal_f1(df)



