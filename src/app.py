import streamlit as st, json, time, joblib, numpy as np, os
from capture import capture_trial, PROMPT
from features import vectorize_trial

st.set_page_config(page_title="Keystroke Auth", page_icon="⌨️")
st.title("Keystroke Dynamics Authentication ⌨️")

tab1, tab2, tab3 = st.tabs(["Enroll", "Verify", "Help"])

with tab1:
    st.header("Enroll (collect your trials)")
    n = st.number_input("Number of trials", 5, 50, 12)
    if st.button("Capture Trials"):
        for i in range(n):
            st.write(f"Trial {i+1}/{n}: console will prompt you.")
            time.sleep(0.5)
            capture_trial(PROMPT, "data/raw/you_genuine.jsonl")
        st.success("Done! Next: collect impostor trials manually.")
        st.info("Train model in terminal: python .\\src\\train.py")

with tab2:
    st.header("Verify your identity")
    if not (os.path.exists("models/clf.joblib") and os.path.exists("models/threshold.npy")):
        st.warning("Train the model first.")
    else:
        if st.button("Capture attempt"):
            capture_trial(PROMPT, "data/raw/tmp.jsonl")
            rec = json.loads(open("data/raw/tmp.jsonl", "r", encoding="utf-8").readlines()[-1])
            clf = joblib.load("models/clf.joblib")
            sc = joblib.load("models/scaler.joblib")
            thr = float(np.load("models/threshold.npy"))
            x = vectorize_trial(rec)[None,:]
            p = clf.predict_proba(sc.transform(x))[0,1]
            st.metric("Score", f"{p:.3f}")
            st.metric("Decision", "ACCEPT" if p >= thr else "REJECT")

with tab3:
    st.header("Help & Ethics")
    st.write("Captures only **timing** (not text), local only, stops when you hit Enter.")
