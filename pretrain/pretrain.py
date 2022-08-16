import pickle
import random
import torch
import math
import numpy as np
import os
import torch.nn as nn
import torch.nn.functional as F
from model import pretrainModel, vocab_size

workloads = ['BiVAE', 'VAECF', 'NCF', 'LightGCN', 'CML', 'CDAE', 'UAutoRec', 'IAutoRec']
workload2codeidx = pickle.load(open('./pretrain/workload2codeidx.pkl', 'rb'))
workload2astidx = pickle.load(open('./pretrain/workload2astidx.pkl', 'rb'))
workload2isType = pickle.load(open('./pretrain/workload2isType.pkl', 'rb'))
w2len = {}
for w in workloads:
    cnt = 0
    for i in workload2codeidx[w]:
        if i != 0:
            cnt += 1
    w2len[w] = cnt


def getData():
    workload = random.choice(workloads)
    test_tokenlen = random.randint(0, w2len[workload])
    token_input = workload2codeidx[workload][0:test_tokenlen]
    token_input = np.concatenate((token_input, [0 for _ in range(300 - test_tokenlen)]))
    token_input = [token_input for _ in range(2)]

    cnt = 0
    id = 0
    for i in workload2isType[workload]:
        if not i:
            cnt += 1
        if cnt == test_tokenlen:
            break
        id += 1
    ast_input = workload2astidx[workload][0:id + 1]
    isType_input = workload2isType[workload][0:id + 1]

    token_label = torch.tensor([workload2codeidx[workload][test_tokenlen]], dtype=torch.long)
    ast_label = torch.tensor([workload2astidx[workload][id + 1]], dtype=torch.long)

    return token_input, torch.from_numpy(ast_input), isType_input, token_label, ast_label

def attention(query, key, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return p_attn

def train():
    model = pretrainModel()
    criterion = nn.CrossEntropyLoss()
    lr = 0.1
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    epochs = 100
    batchsize = 32
    for epoch in range(epochs):
        loss = torch.tensor(0, dtype=torch.float)
        for i in range(batchsize):
            token_input, ast_input, isType_input, token_label, ast_label = getData()
            _, token_pred, type_pred = model(token_input, ast_input, isType_input, token_label, ast_label)
        loss += criterion(F.softmax(token_pred.reshape(1, vocab_size)), token_label) + criterion(F.softmax(type_pred.reshape(1, vocab_size)), ast_label)
        loss /= batchsize*2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with open(os.path.join("./pretrain/pretrain_model") + ".pth", "wb") as f:
            torch.save(model.state_dict(), f)


def eval():
    w2pretrainfeat = {}
    model = pretrainModel()
    model.load_state_dict(torch.load("./pretrain/pretrain_model.pth"))
    for (w, tokenlen) in w2len.items():
        token_input = workload2codeidx[w]
        token_input = [token_input for _ in range(2)]
        astlen = len(workload2isType[w])
        ast_input = torch.from_numpy(workload2astidx[w][0:astlen])
        isType_input = workload2isType[w]
        token_label = torch.tensor([0])
        ast_label = torch.tensor([0])
        ast_oup, _, _ = model(token_input, ast_input, isType_input, token_label, ast_label)
        w2pretrainfeat[w] = torch.mean(ast_oup, 0).detach().numpy()

    k = np.zeros((len(workloads), 128))
    for i in range(len(workloads)):
        w = workloads[i]
        k[i, :] = w2pretrainfeat[w]
    att = attention(torch.tensor(k), torch.tensor(k)).detach().numpy()
    w2attention = {}
    for i in range((len(workloads))):
        w1 = workloads[i]
        w2attention[w1] = {}
        for j in range((len(workloads))):
            w2 = workloads[j]
            w2attention[w1][w2] = att[i][j]

    pickle.dump(w2pretrainfeat, open("./data/w2pretrainfeat.pkl", 'wb'))
    pickle.dump(w2attention, open("./data/w2attention_pretrain.pkl", 'wb'))

if __name__ == '__main__':
    # train()
    eval()