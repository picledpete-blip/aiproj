import json
import keras
import numpy as np
from tqdm import tqdm
mpath = "p1.keras"
with open("toks.json", "r") as file:
    toks = json.load(file)
toks = {token: i for i, token in enumerate(toks.keys())}
def idchar(id):
    for k, v in toks.items():
        if v == id:
            return k
    return ""

maxsize = 4
def nextid(text, pos):
    chunk = text[pos:pos + maxsize]

    for i in range(len(chunk), 0, -1):
        oid = toks.get(chunk[:i])

        if oid is not None:
            return oid, i

    return 0, 1

def genids(text):
    out = []
    pos = 0
    while pos < len(text):
        token_id, length = nextid(text, pos)
        out.append(token_id)
        pos += length
    return out


context_size = 128




model = keras.Sequential([
    keras.layers.Input(shape=(context_size,)),

    keras.layers.Embedding(
        input_dim=len(toks),
        output_dim=64
    ),

    
    keras.layers.LSTM(128, return_sequences=True),
    #keras.layers.LSTM(128, return_sequences=True),
    keras.layers.LSTM(128, return_sequences=True),
    keras.layers.GRU(128, return_sequences=True),
    keras.layers.GRU(128),
    #keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(128, activation="relu"),

    keras.layers.Dense(
        len(toks),
        activation="softmax"
    )
])
try:
    model = keras.models.load_model(mpath)
except:
    pass

train = []
with open("pg10.txt", "r", encoding="utf-8") as f:
    text = f.read()
    
    pos = 0
    with tqdm(total=len(text), desc="Generating ids") as pbar:
        while pos < len(text):
            token_id, length = nextid(text, pos)        
            train.append(token_id)
            pos += length
            pbar.update(1)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",

)

print("Largest training ID:", max(train))
print("Model vocabulary size:", len(toks))

assert min(train) >= 0
assert max(train) < len(toks)

x = []
y = []
for i in tqdm(range(context_size, len(train)), desc= "Preparing Data"):
    x.append(train[i - context_size:i])
    y.append(train[i])

x = np.array(x, dtype=np.int32)
y = np.array(y, dtype=np.int32)
checkpoint = keras.callbacks.ModelCheckpoint(
    filepath="p1.keras",
    save_best_only=False,
    save_weights_only=False,
    save_freq=500,
    verbose=1
)

model.fit(x, y, epochs=1, batch_size=32, callbacks=[checkpoint])

model.save(mpath)
