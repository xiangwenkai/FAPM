import pandas as pd
from utils import Ontology


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

godb = Ontology(f'/cluster/home/wenkai/LAVIS/data/go1.4-basic.obo', with_rels=True)

case_mf = pd.read_csv('/cluster/home/wenkai/LAVIS/data/pretrain/cases_mf.csv', sep='|')

# bp case, 包括辣椒受体
case_bp = pd.read_csv('/cluster/home/wenkai/LAVIS/data/pretrain/cases_bp.csv', sep='|')
case_bp['GO_label'] = case_bp['GO_label'].apply(lambda x: [i.strip() for i in x.split(';')])
case_bp = prop(case_bp)
case_bp['GO_label'] = case_bp['GO_label'].apply(lambda x: '; '.join(x))
case_bp['prop_annotations'] = case_bp['prop_annotations'].apply(lambda x: '; '.join(x))
case_bp[['name', 'protein', 'function', 'GO_label', 'id', 'prompt', 'prop_annotations']].to_pickle('/cluster/home/wenkai/deepgo2/data/bp/cases_data.pkl')

case_mf['GO_label'] = case_mf['GO_label'].apply(lambda x: [i.strip() for i in x.split(';')])
case_mf = prop(case_mf)
case_mf['GO_label'] = case_mf['GO_label'].apply(lambda x: '; '.join(x))
case_mf['prop_annotations'] = case_mf['prop_annotations'].apply(lambda x: '; '.join(x))

case_bp['GO_label'] = case_bp['GO_label'].apply(lambda x: [i.strip() for i in x.split(';')])
case_bp = prop(case_bp)
case_mf[['name', 'protein', 'function', 'GO_label', 'id', 'prompt', 'prop_annotations']].to_pickle('/cluster/home/wenkai/deepgo2/data/mf/cases_data_445772.pkl')













