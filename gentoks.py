from itertools import islice
import json
import string
from collections import Counter

maxsize = 4
valid = string.ascii_letters + string.digits + string.punctuation + " " + "\n"

with open("pg10.txt", "r", encoding="utf-8") as f:
    text = list(islice(f, 50000))

toksa = sorted(set(" ".join(text)))
toks = {}
cc = 0
for char in toksa:
    if char in valid:
        toks[char] = cc
        cc += 1
words = []
for p in " ".join(text).split(" "):
    words.append(p)

twords = [item for item, _ in Counter(words).most_common(250)]
for word in twords:
    try:
        word = word[:maxsize]
    except:
        pass
    if not all(c in valid for c in word):
        continue
    toks[word] = cc
    cc += 1
toks = {token: i for i, token in enumerate(toks.keys())}
with open("toks.json", "w") as file:
    json.dump(toks, file, indent=1)