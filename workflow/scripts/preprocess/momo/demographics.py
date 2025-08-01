import pandas as pd 
import sqlite3 
import numpy as np

ROOT = '/m/cs/scratch/networks-nima-mmm2018/data/mmm2018'
groups = ['mmm-control', 'mmm-bd', 'mmm-bpd', 'mmm-mdd']

# Utility
def score_neo(df, id_col="user"):
    
    # domains
    N = [1,3,11,14,25,33,42,43,47,56,60,67]
    E = [2,4,7,8,12,17,18,19,22,23,24,26,29,41,50,52,53,54,59]
    A = [15,28,32,36,39,40,46,48,49,55,58,66]
    C = [9,10,13,16,21,35,38,45,51,57,63,64]
    O = [5,6,20,27,30,31,34,37,44,61,62,65]
    rev = {2,12,14,20,22,24,25,29,33,37,39,40,41,44,47,48,49,50,54,55,57,61,62,65}

    # To int
    df['answer'] = df['answer'].astype(int)
    
    df["item"] = df["id"].str.extract(r"(\d+)").astype(int)

    df["keyed"] = np.where(df["item"].isin(rev), 6 - df["answer"], df["answer"])

    wide = df.pivot_table(index=id_col, columns="item", values="keyed", aggfunc="last")

    def _sum(items):
        present = [i for i in items if i in wide.columns]
        return wide[present].sum(axis=1, min_count=1)

    scores = pd.DataFrame({
        "Neuroticism":       _sum(N),
        "Extraversion":      _sum(E),
        "Agreeableness":     _sum(A),
        "Conscientiousness": _sum(C),
        "Openness":          _sum(O),
    })
    
    return scores
demos, neos = [], []

for group in groups:
    
    # Basic demographics    
    db_path = f'{ROOT}/{group}/MMMBackgroundAnswers.sqlite3'

    with sqlite3.connect(db_path) as conn:
       
        demo = pd.read_sql("SELECT * FROM MMMBackgroundAnswers;", conn)
        demo = demo.pivot_table(index=['user'], columns='id', values='answer', aggfunc='first').reset_index()
        demo["group"] = group  # keep track of source group
        demos.append(demo)

        

    # NEO 
    neo_path = f'{ROOT}/{group}/MMMPostActiveAnswers.sqlite3'
    with sqlite3.connect(neo_path) as conn:
        baseline = pd.read_sql("SELECT * FROM MMMPostActiveAnswers;", conn)

        # Keep only NEO score
        baseline = baseline[baseline.id.str.startswith('NEO')]
        
        score = score_neo(baseline, id_col="user")
        score['group'] = group
        neos.append(score)

# Concat, merge
demos = pd.concat(demos)
neos = pd.concat(neos)

df = demos.merge(neos, on=['user', 'group'])

# To csv
OUT_DIR = '/m/cs/work/luongn1/digirhythm-v2/data/interim/momo/demographics_neo.csv'