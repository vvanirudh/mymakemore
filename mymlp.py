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
W1 = torch.randn((context_embed_size, 200), generator=g) * (5/3) / (context_embed_size**0.5)
b1 = torch.randn(200, generator=g) * 0.01
W2 = torch.randn((200, 27), generator=g) * 0.01
b2 = torch.randn(27, generator=g) * 0.0
bngain = torch.ones((1, 200))
bnbias = torch.zeros((1, 200))
bnmean_running = torch.zeros((1, 200))
bnstd_running = torch.ones((1, 200))
parameters = [C, W1, b1, W2, b2, bngain, bnbias]
for p in parameters:
    p.requires_grad = True

# lre = torch.linspace(-3, 0, 1000)
# lrs = 10**lre

# lri = []
lossi = []
for i in range(200000):
    # minibatch construct
    ix = torch.randint(0, Xtr.shape[0], (32,))
    X_batch = Xtr[ix]
    Y_batch = Ytr[ix]
    # forward pass
    # (N, 3, 2)
    emb = C[X_batch]
    hpreact = emb.view(-1, context_embed_size) @ W1 # + b1 # bias is not needed as norm layer removes it
    # apply batch normalization
    bnmeani = hpreact.mean(0, keepdim=True)
    bnstdi = hpreact.std(0, keepdim=True) + 1e-8
    hpreact = bngain * (hpreact - bnmeani) / bnstdi + bnbias
    with torch.no_grad():
        bnmean_running = 0.999 * bnmean_running + 0.001 * bnmeani
        bnstd_running = 0.999 * bnstd_running + 0.001 * bnstdi
    h = torch.tanh(hpreact)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Y_batch)
    # print(f"{loss=}")

    # backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    # lr = lrs[i]
    lr = 10**-1
    if i > 100000:
        lr = 10**-2
    for p in parameters:
        p.data += -lr * p.grad

    # lri.append(lre[i])
    lossi.append(loss.log10().item())

# plt.hist(hpreact.view(-1).tolist(), 50)
# plt.show()

# plt.figure(figsize=(20, 10))
# plt.imshow(h.abs() > 0.99, cmap="gray", interpolation="nearest")
# plt.show()

# plt.plot(lri, lossi)
# plt.show()
# plt.savefig('lrs.png')

# plt.plot(range(len(lossi)), lossi)
# plt.show()
# plt.savefig('loss.png')

# Calibrate batchnorm at end of training
# with torch.no_grad():
#     # pass the training set through
#     emb = C[Xtr]
#     embcat = emb.view(emb.shape[0], -1)
#     hpreact = embcat @ W1 + b1
#     bnmean = hpreact.mean(0, keepdim=True)
#     bnstd = hpreact.std(0, keepdim=True)

# Evaluate on train dataset
with torch.no_grad():
    emb = C[Xtr]
    hpreact = emb.view(-1, context_embed_size) @ W1 + b1
    # apply batch normalization
    hpreact = bngain * (hpreact - bnmean_running) / bnstd_running + bnbias
    h = torch.tanh(hpreact)
    logits = h @ W2 + b2
    train_loss = F.cross_entropy(logits, Ytr)
    print(f"{train_loss=}")

# Evaluate on validation dataset
with torch.no_grad():
    emb = C[Xdev]
    hpreact = emb.view(-1, context_embed_size) @ W1 + b1
    # apply batch normalization
    hpreact = bngain * (hpreact - bnmean_running) / bnstd_running + bnbias
    h = torch.tanh(hpreact)
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
        hpreact = emb.view(-1, context_embed_size) @ W1 + b1
        # apply batch normalization
        hpreact = bngain * (hpreact - bnmean_running) / bnstd_running + bnbias
        h = torch.tanh(hpreact)
        logits = h @ W2 + b2
        p = F.softmax(logits, dim=1)
        ix = torch.multinomial(p, 1, generator=g).item()
        out.append(itos[ix])
        context = context[1:] + [ix]
        if ix == 0:
            break
    print(''.join(out))
