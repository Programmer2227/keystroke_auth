import numpy as np

def pad(arr, L):
    arr = np.array(arr, float)
    if len(arr) >= L: return arr[:L]
    return np.pad(arr, (0, L-len(arr)))

def dwell_flight(events):
    downs, ups = [], []
    for ev in events:
        if ev["e"] == "down": downs.append(ev["t"])
        elif ev["e"] == "up": ups.append(ev["t"])
    n = min(len(downs), len(ups))
    dwell = [max(ups[i]-downs[i], 0) for i in range(n)]
    flights = [max(downs[i+1]-ups[i], 0) for i in range(n-1)]
    return np.array(dwell), np.array(flights), np.array(downs), np.array(ups)

def aggregates(dwell, flights):
    def stats(x):
        if len(x)==0: return [0.0, 0.0, 0.0]
        return [float(np.mean(x)), float(np.std(x)), float(np.median(x))]
    speed = 0.0 if len(dwell)==0 else (len(dwell)/(np.sum(dwell)+1e-6))
    return np.array(stats(dwell) + stats(flights) + [speed], dtype=float)

def vectorize_trial(rec, Kd=60, Kf=59):
    dwell, flights, _, _ = dwell_flight(rec["events"])
    d = pad(dwell, Kd)
    f = pad(flights, Kf)
    agg = aggregates(dwell, flights)
    return np.hstack([d, f, agg])
