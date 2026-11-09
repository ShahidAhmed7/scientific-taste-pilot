import json, numpy as np, collections, random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import vstack
def load(f):
    out=[]
    for l in open(f):
        try: out.append(json.loads(l))
        except Exception: pass
    return out
test, year = load('test.jsonl'), load('test_ood_year.jsonl')
tr = load('train_part.jsonl')
print('train categories', collections.Counter(r['paper_a_category'] for r in tr).most_common(), ' test', collections.Counter(r['paper_a_category'] for r in test).most_common())
held = {r[f'paper_{s}_arxiv_id'] for d in (test,year) for r in d for s in 'ab'}
tr = [r for r in tr if r['paper_a_arxiv_id'] not in held and r['paper_b_arxiv_id'] not in held]
random.seed(0); random.shuffle(tr)
text = lambda r,s: r[f'paper_{s}_title']+' . '+r[f'paper_{s}_abstract']
lab = lambda d: np.array([r['correct_answer']=='A' for r in d])
for n in (25000, 100000, len(tr)):
    sub = tr[:n]
    vec = TfidfVectorizer(min_df=3, max_features=200000, ngram_range=(1,2), sublinear_tf=True, dtype=np.float32).fit([text(r,'a') for r in sub[:60000]]+[text(r,'b') for r in sub[:60000]])
    F = lambda d: (vec.transform([text(r,'a') for r in d]) - vec.transform([text(r,'b') for r in d])).tocsr()
    X = F(sub); y = lab(sub)
    clf = LogisticRegression(C=1, max_iter=2000, solver='liblinear').fit(vstack([X,-X]), np.concatenate([y,~y]))
    acc = [100*np.mean(clf.predict(F(d))==lab(d)) for d in (test,year)]
    bycat = collections.defaultdict(list)
    for r,p in zip(test, clf.predict(F(test))): bycat[r['paper_a_category']].append(p==(r['correct_answer']=='A'))
    print(f"n={n}: main {acc[0]:.1f}  future-year {acc[1]:.1f}  by field: " + ', '.join(f"{k} {100*np.mean(v):.1f} ({len(v)})" for k,v in bycat.items()))
