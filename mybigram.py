import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

words = open("names.txt", "r").read().splitlines()

## BIGRAM MODEL
b = {}
for w in words:
    chs = ['<S>'] + list(w) + ['<E>']
    for ch1, ch2 in zip(chs, chs[1:]):
        bigram = (ch1, ch2)
        b[bigram] = b.get(bigram, 0) + 1

# Count bigrams
N = torch.zeros((27,27), dtype=torch.int32)
chars = ['.'] + sorted(list(set(''.join(words))))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1

# Visualize bigram counts
plt.figure(figsize=(10, 10))
plt.imshow(N, cmap='Blues')
for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha='center', va='bottom', color='gray')
        plt.text(j, i, str(N[i, j].item()), ha='center', va='top', color='gray')
plt.axis('off')
# plt.show()
plt.savefig('bigram.png')

# Sample words from bigram model
g = torch.Generator().manual_seed(0)
# Add psuedo counts -- model smoothing
P = (N + 1).float()
P /= P.sum(dim=1, keepdim=True)

for _ in range(5):
    ix = 0
    word = ''
    while True:
        p = P[ix]
        ix = torch.multinomial(p, 1, generator=g).item()
        if ix == 0:
            break
        word += itos[ix]
    # print(word)

# Likelihood computation
log_likelihood = 0.0
count = 0
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1, ix2]
        logprob = torch.log(prob).item()
        log_likelihood += logprob
        count += 1
        # print(f"{ch1}{ch2}: {logprob:.4f}")
nll = -log_likelihood / count
print(f"{nll=}")

# Create training set
xs, ys = [], []
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        xs.append(stoi[ch1])
        ys.append(stoi[ch2])

xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()

# one hot encoding
xenc = F.one_hot(xs, num_classes=27).float()
yenc = F.one_hot(ys, num_classes=27).float()
plt.clf()
plt.imshow(xenc)
# plt.show()
plt.savefig('onehot.png')

# Define neural network
W = torch.randn((27, 27), generator=g)
logits = xenc @ W
counts = logits.exp()
probs = counts / counts.sum(dim=1, keepdim=True)
# print(probs)

nlls = torch.zeros(num)
for i in range(num):
    x = xs[i].item()
    y = ys[i].item()
    p = probs[i, y]
    logp = torch.log(p)
    nlls[i] = -logp
# print(nlls.mean().item())

# Optimization
W = torch.randn((27, 27), requires_grad=True, generator=g)

for _ in range(1000):
    ## forward pass
    logits = xenc @ W
    counts = logits.exp()  # This is a proxy for N above
    probs = counts / counts.sum(dim=1, keepdim=True)
    loss = -probs[torch.arange(num), ys].log().mean()
    loss += (W**2).mean() * 0.01 # L2 regularization
    print(f"{loss=}")
    ## backward pass
    W.grad = None # set the gradient to zero
    loss.backward()
    ## gradient descent
    W.data += -50.0 * W.grad

# Sample from the model
g = torch.Generator().manual_seed(0)
for _ in range(5):
    ix = 0
    word = ''
    while True:
        xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
        logits = xenc @ W
        counts = logits.exp()
        p = counts / counts.sum(dim=1, keepdim=True)
        ix = torch.multinomial(p, 1, generator=g).item()
        if ix == 0:
            break
        word += itos[ix]
    print(word)
