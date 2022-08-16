import torch
import math, copy
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from TransformerEnc import make_transformer_encoder


vocab_size = 323
d_model = 512

code_vocab_file = open('./pretrain/vocab.code', encoding='utf-8')
freq = [0, 0, 0, 0]
for v in code_vocab_file:
  v = v.split('\t')[1]
  freq.append(int(v))
freq = torch.FloatTensor(freq)

class pretrainModel(nn.Module):
    @staticmethod
    def weight_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)

    def __init__(self):
        super(pretrainModel, self).__init__()
        self.transformerEncoder_layer = make_transformer_encoder(vocab_size)

        self.ast_embedding_layer=nn.Embedding(vocab_size, d_model)
        self.lstm_layer = nn.LSTM(d_model, hidden_size=128)

        self.token_mlp_layer = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, vocab_size)
            )

        self.type_mlp_layer = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, vocab_size)
        )

        self.apply(self.weight_init)


    def forward(self, token_input, ast_input, isType_input, token_label, ast_label):
        token_oup = self.transformerEncoder_layer(token_input)
        # print("transformer output", token_oup.shape)
        ast_emb = self.ast_embedding_layer(ast_input)
        # print("ast embedding", ast_emb.shape)
        id = 0
        for i in range(len(isType_input)):
            if isType_input == False:
                ast_emb[i] = token_oup[0][id]
                id += 1
        ast_emb = ast_emb.view(len(ast_emb), 1, -1)
        ast_oup, _ = self.lstm_layer(ast_emb)
        # print("ast lstm oup", ast_oup.shape)
        token_pred = self.token_mlp_layer(ast_oup[-1])
        type_pred = self.type_mlp_layer(ast_oup[-1])

        neg_sample = torch.multinomial(freq, 20)
        mask = torch.ones(vocab_size, dtype=torch.bool)
        mask = mask.scatter(0, neg_sample, False)
        mask = mask.scatter(0, token_label, False)
        mask = mask.scatter(0, ast_label, False)

        masked_token_pred = token_pred.masked_fill(mask, -np.inf)
        masked_type_pred = type_pred.masked_fill(mask, -np.inf)

        return ast_oup, masked_token_pred, masked_type_pred
