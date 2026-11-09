import json, numpy as np, datetime as dt, re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
def load(f): return [json.loads(l) for l in open(f)]
test, year, ext = load('test.jsonl'), load('test_ood_year.jsonl'), load('test_extended.jsonl')
def pid(r,s): return r[f'paper_{s}_arxiv_id']
held = {pid(r,s) for d in (test,year) for r in d for s in 'ab'}
train = [r for r in ext if pid(r,'a') not in held and pid(r,'b') not in held]
print(f"train pairs after removing overlap: {len(train)} of {len(ext)}; label balance test A-share={np.mean([r['correct_answer']=='A' for r in test]):.3f}")
def date(r,s): return dt.datetime.fromisoformat(str(r[f'paper_{s}_date'])[:10])
def heur(d, f):
    acc=[]
    for r in d:
        a,b=f(r,'a'),f(r,'b')
        pred = 'A' if a>b else 'B' if b>a else None
        acc.append(0.5 if pred is None else float(pred==r['correct_answer']))
    return 100*np.mean(acc)
older = lambda r,s: -date(r,s).timestamp()
abslen = lambda r,s: len(r[f'paper_{s}_abstract'].split())
titlen = lambda r,s: len(r[f'paper_{s}_title'].split())
for name,d in (('main test',test),('future-year',year)):
    gaps=[abs((date(r,'a')-date(r,'b')).days) for r in d]
    print(f"{name}: n={len(d)}  older-wins={heur(d,older):.1f}  longer-abstract={heur(d,abslen):.1f}  longer-title={heur(d,titlen):.1f}  median date gap={np.median(gaps):.0f} days")
text = lambda r,s: r[f'paper_{s}_title']+' . '+r[f'paper_{s}_abstract']
vec = TfidfVectorizer(min_df=3, max_features=50000, ngram_range=(1,2), sublinear_tf=True).fit([text(r,s) for r in train for s in 'ab'])
def feats(d, flip=False):
    A = vec.transform([text(r,'a') for r in d]); B = vec.transform([text(r,'b') for r in d])
    extra = np.array([[ (date(r,'b')-date(r,'a')).days/365, np.log1p(abslen(r,'a'))-np.log1p(abslen(r,'b')), titlen(r,'a')-titlen(r,'b')] for r in d])
    from scipy.sparse import hstack, csr_matrix
    return hstack([A-B, csr_matrix(extra)]).tocsr()
# symmetrize training: each pair in both orders
X = feats(train); y = np.array([r['correct_answer']=='A' for r in train])
from scipy.sparse import vstack
Xs, ys = vstack([X, -X]), np.concatenate([y, ~y])
for C in (0.3,1,3):
    clf = LogisticRegression(C=C, max_iter=3000).fit(Xs, ys)
    res = []
    for d in (test, year):
        p = clf.predict(feats(d)); res.append(100*np.mean(p == np.array([r['correct_answer']=='A' for r in d])))
    print(f"TF-IDF+date+length logistic (C={C}): main test {res[0]:.1f}  future-year {res[1]:.1f}")
# date-free variant to separate the date cue
def feats_nodate(d):
    from scipy.sparse import hstack, csr_matrix
    A = vec.transform([text(r,'a') for r in d]); B = vec.transform([text(r,'b') for r in d])
    return (A-B).tocsr()
X2 = feats_nodate(train); clf2 = LogisticRegression(C=1, max_iter=3000).fit(vstack([X2,-X2]), ys)
print("TF-IDF words only (no date, no length): main", round(100*np.mean(clf2.predict(feats_nodate(test))==np.array([r['correct_answer']=='A' for r in test])),1),
      " future-year", round(100*np.mean(clf2.predict(feats_nodate(year))==np.array([r['correct_answer']=='A' for r in year])),1))
coef = clf2.coef_[0]; names = np.array(vec.get_feature_names_out())
print("top 'more cited' words:", ', '.join(names[np.argsort(-coef)[:30]]))
print("top 'less cited' words:", ', '.join(names[np.argsort(coef)[:30]]))
