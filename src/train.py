import json, numpy as np, joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from features import vectorize_trial

def load_jsonl(path):
    X = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            X.append(json.loads(line))
    return X

def main():
    you = load_jsonl("data/raw/you_genuine.jsonl")
    imp = load_jsonl("data/raw/impostors.jsonl")

    X, y = [], []
    for rec in you:
        X.append(vectorize_trial(rec)); y.append(1)
    for rec in imp:
        X.append(vectorize_trial(rec)); y.append(0)

    X = np.vstack(X); y = np.array(y)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)

    clf = LogisticRegression(max_iter=400, class_weight="balanced")
    clf.fit(Xtr_s, ytr)

    proba = clf.predict_proba(Xte_s)[:,1]
    auc = roc_auc_score(yte, proba)
    fpr, tpr, thr = roc_curve(yte, proba)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer_thr = thr[eer_idx]
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2

    print(f"ROC-AUC={auc:.3f}  EER={eer:.2%}")

    Path("models").mkdir(exist_ok=True)
    joblib.dump(clf, "models/clf.joblib")
    joblib.dump(sc, "models/scaler.joblib")
    np.save("models/threshold.npy", eer_thr)

if __name__ == "__main__":
    main()
