import keras
import json
model = keras.models.load_model("p1.keras")
context_size = 128
with open("toks.json", "r") as file:
    toks = json.load(file)


def idchar(id):
    for k, v in toks.items():
        if v == id:
            return k
    return ""
maxsize = 4
def nextid(char):

    for i in range(maxsize, 0, -1):
            oid = toks.get(char[:i])
            if oid is not None:
                return [oid, i]

    return [0,1]

def genids(text):
    out = []
    while len(text) > 0:
            nid = nextid(text)
            text = text[nid[1]:]
            out.append(nid[0])
    return out

import numpy as np

def generate(seed, length=100,show=False):
    result = seed
    if show:
        print(result, end="", flush=True)
    for _ in range(length):
        context = result[-context_size:]
        ids = genids(context)

        ids = [0] * (context_size - len(ids)) + ids
        
        probs = model.predict(
            np.array([ids], dtype=np.int32),
            verbose=0
        )[0]
        
        next_id = np.random.choice(len(toks), p=probs)
        result += idchar(next_id)
        if show:
            print(idchar(next_id), end="", flush=True)
    return result

generate("", length=300, show=True)