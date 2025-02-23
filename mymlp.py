import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn

words = open("names.txt", "r").read().splitlines()

chars = ['.'] + sorted(list(set(''.join(words))))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# Build the dataset
block_size = 3 # context length
def build_dataset(words):
    X, Y = [], []
    for w in words:
        # print(w)
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            # print(''.join(itos[i] for i in context), "--->", ch)
            context = context[1:] + [ix] # Slide context

    return torch.tensor(X), torch.tensor(Y)

import random
random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))
Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])
g = torch.Generator().manual_seed(0)

# Model
embed_size = 10
context_embed_size = block_size * embed_size
C = torch.randn((27, embed_size), generator=g)
W1 = torch.randn((context_embed_size, 200), generator=g)
b1 = torch.randn(200, generator=g)
W2 = torch.randn((200, 27), generator=g)
b2 = torch.randn(27, generator=g)
parameters = [C, W1, b1, W2, b2]
for p in parameters:
    p.requires_grad = True

# lre = torch.linspace(-3, 0, 1000)
# lrs = 10**lre

# lri = []
lossi = []
for i in range(100000):
    # minibatch construct
    ix = torch.randint(0, Xtr.shape[0], (32,))
    X_batch = Xtr[ix]
    Y_batch = Ytr[ix]
    # forward pass
    # (N, 3, 2)
    emb = C[X_batch]
    h = torch.tanh(emb.view(-1, context_embed_size) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Y_batch)
    # print(f"{loss=}")

    # backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    # lr = lrs[i]
    lr = 10**-1
    if i > 20000:
        lr = 10**-2
    elif i > 60000:
        lr = 5 * 10**-3
    for p in parameters:
        p.data += -lr * p.grad

    # lri.append(lre[i])
    lossi.append(loss.log10().item())

# plt.plot(lri, lossi)
# plt.show()
# plt.savefig('lrs.png')

# plt.plot(range(len(lossi)), lossi)
# plt.show()
# plt.savefig('loss.png')

# Evaluate on train dataset
emb = C[Xtr]
h = torch.tanh(emb.view(-1, context_embed_size) @ W1 + b1)
logits = h @ W2 + b2
train_loss = F.cross_entropy(logits, Ytr)
print(f"{train_loss=}")

# Evaluate on validation dataset
emb = C[Xdev]
h = torch.tanh(emb.view(-1, context_embed_size) @ W1 + b1)
logits = h @ W2 + b2
val_loss = F.cross_entropy(logits, Ydev)
print(f"{val_loss=}")

# plt.figure(figsize=(8, 8))
# plt.scatter(C[:, 0].data, C[:, 1].data, s=200)
# for i in range(C.shape[0]):
#     plt.text(C[i, 0].item(), C[i, 1].item(), itos[i], ha='center', va='center', color='white')
# plt.grid('minor')
# plt.show()

# Sample from the model
g = torch.Generator().manual_seed(0)
for _ in range(20):
    out = []
    context = [0] * block_size
    while True:
        emb = C[torch.tensor(context)]
        h = torch.tanh(emb.view(-1, context_embed_size) @ W1 + b1)
        logits = h @ W2 + b2
        p = F.softmax(logits, dim=1)
        ix = torch.multinomial(p, 1, generator=g).item()
        out.append(itos[ix])
        context = context[1:] + [ix]
        if ix == 0:
            break
    print(''.join(out))
