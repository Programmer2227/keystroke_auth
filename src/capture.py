from pynput import keyboard
import time, json, os, sys

PROMPT = "the quick brown fox jumps over 13 lazy dogs"

def capture_trial(prompt: str, out_path: str):
    print(f"Type this exactly, then press Enter:\n{prompt}\n")
    buf = []
    t0 = time.perf_counter()

    def on_press(key):
        try: k = key.char.lower()
        except: k = str(key)
        buf.append({"k": k, "e": "down", "t": time.perf_counter() - t0})

    def on_release(key):
        try: k = key.char.lower()
        except: k = str(key)
        buf.append({"k": k, "e": "up", "t": time.perf_counter() - t0})
        if key == keyboard.Key.enter:
            return False  # stop listening

    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"prompt": prompt, "events": buf}) + "\n")
    print(f"Saved one trial to {out_path}")

if __name__ == "__main__":
    out = "data/raw/you_genuine.jsonl" if len(sys.argv) < 2 else sys.argv[1]
    capture_trial(PROMPT, out)
