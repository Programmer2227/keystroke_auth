import sys, json, numpy as np, joblib
from features import vectorize_trial

clf = joblib.load("models/clf.joblib")
sc  = joblib.load("models/scaler.joblib")
thr = float(np.load("models/threshold.npy"))

def score_record(rec):
    x = vectorize_trial(rec)[None,:]
    xs = sc.transform(x)
    p = clf.predict_proba(xs)[0,1]
    return float(p), bool(p >= thr)

if __name__ == "__main__":
    rec = json.loads(sys.stdin.read())
    p, ok = score_record(rec)
    print({"score": p, "accept": ok})
