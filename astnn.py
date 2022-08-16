import json
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import numpy as np
from gensim.models.word2vec import Word2Vec


class BatchTreeEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, encode_dim, use_gpu, pretrained_weight=None):
        super(BatchTreeEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encode_dim = encode_dim
        self.W_c = nn.Linear(embedding_dim, encode_dim)
        self.W_l = nn.Linear(encode_dim, encode_dim)
        self.W_r = nn.Linear(encode_dim, encode_dim)
        self.activation = F.relu
        self.stop = -1
        self.batch_size = None
        self.use_gpu = use_gpu
        self.node_list = []
        self.th = torch.cuda if use_gpu else torch
        self.batch_node = None
        # pretrained  embedding
        if pretrained_weight is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))
            # self.embedding.weight.requires_grad = False

    def create_tensor(self, tensor):
        if self.use_gpu:
            return tensor.cuda()
        return tensor

    def traverse_mul(self, node, batch_index):
        size = len(node)
        if not size:
            return None
        batch_current = self.create_tensor(Variable(torch.zeros(size, self.encode_dim)))

        index, children_index = [], []
        current_node, children = [], []
        for i in range(size):
            if node[i][0] is not -1:
                index.append(i)
                current_node.append(node[i][0])
                temp = node[i][1:]
                c_num = len(temp)
                for j in range(c_num):
                    if temp[j][0] is not -1:
                        if len(children_index) <= j:
                            children_index.append([i])
                            children.append([temp[j]])
                        else:
                            children_index[j].append(i)
                            children[j].append(temp[j])
            else:
                batch_index[i] = -1

        batch_current = self.W_c(batch_current.index_copy(0, Variable(self.th.LongTensor(index)),
                                                          self.embedding(Variable(self.th.LongTensor(current_node)))))

        for c in range(len(children)):
            zeros = self.create_tensor(Variable(torch.zeros(size, self.encode_dim)))
            batch_children_index = [batch_index[i] for i in children_index[c]]
            tree = self.traverse_mul(children[c], batch_children_index)
            if tree is not None:
                batch_current += zeros.index_copy(0, Variable(self.th.LongTensor(children_index[c])), tree)
        # batch_current = F.tanh(batch_current)
        batch_index = [i for i in batch_index if i is not -1]
        b_in = Variable(self.th.LongTensor(batch_index))
        self.node_list.append(self.batch_node.index_copy(0, b_in, batch_current))
        return batch_current

    def forward(self, x, bs):
        self.batch_size = bs
        self.batch_node = self.create_tensor(Variable(torch.zeros(self.batch_size, self.encode_dim)))
        self.node_list = []
        self.traverse_mul(x, list(range(self.batch_size)))
        self.node_list = torch.stack(self.node_list)
        return torch.max(self.node_list, 0)[0]


class BatchProgramEncoder(nn.Module):
    # def __init__(self, embedding_dim, hidden_dim, vocab_size, encode_dim, label_size, batch_size, use_gpu=True, pretrained_weight=None):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, encode_dim, use_gpu=True, pretrained_weight=None):
        super(BatchProgramEncoder, self).__init__()
        self.stop = [vocab_size-1]
        self.hidden_dim = hidden_dim
        self.num_layers = 1
        self.gpu = use_gpu
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.encode_dim = encode_dim
        #class "BatchTreeEncoder"
        self.encoder = BatchTreeEncoder(self.vocab_size, self.embedding_dim, self.encode_dim,
                                        self.gpu, pretrained_weight)
        # self.root2label = nn.Linear(self.encode_dim, self.label_size)
        # gru
        self.bigru = nn.GRU(self.encode_dim, self.hidden_dim, num_layers=self.num_layers, bidirectional=True,
                            batch_first=True)
        # linear
        # self.hidden2label = nn.Linear(self.hidden_dim * 2, self.label_size)
        # hidden
        self.dropout = nn.Dropout(0.2)

    def get_zeros(self, num):
        zeros = Variable(torch.zeros(num, self.encode_dim))
        if self.gpu:
            return zeros.cuda()
        return zeros

    def forward(self, x):
        batch_size = len(x)
        lens = [len(item) for item in x]
        max_len = max(lens)
        encodes = []
        for i in range(batch_size):
            for j in range(lens[i]):
                encodes.append(x[i][j])

        encodes = self.encoder(encodes, sum(lens))
        seq, start, end = [], 0, 0
        for i in range(batch_size):
            end += lens[i]
            if max_len-lens[i]:
                seq.append(self.get_zeros(max_len-lens[i]))
            seq.append(encodes[start:end])
            start = end
        encodes = torch.cat(seq)
        encodes = encodes.view(batch_size, max_len, -1)

        # gru
        gru_out, hidden = self.bigru(encodes)

        gru_out = torch.transpose(gru_out, 1, 2)
        # pooling
        gru_out = F.max_pool1d(gru_out, gru_out.size(2)).squeeze(2)
        # gru_out = gru_out[:,-1]

        # linear
        # y = self.hidden2label(gru_out)
        return gru_out


def build_astnn():
    emb_size = 128
    emb_path = 'data/ast/node_w2v_' + str(emb_size)
    word2vec = Word2Vec.load(emb_path).wv
    embeddings = np.zeros((word2vec.vectors.shape[0] + 1, word2vec.vectors.shape[1]), dtype="float32")
    embeddings[:word2vec.vectors.shape[0]] = word2vec.vectors

    HIDDEN_DIM = 32
    ENCODE_DIM = 128
    USE_GPU = False
    MAX_TOKENS = word2vec.vectors.shape[0]
    EMBEDDING_DIM = word2vec.vectors.shape[1]

    model = BatchProgramEncoder(EMBEDDING_DIM, HIDDEN_DIM, MAX_TOKENS + 1, ENCODE_DIM,
                                   USE_GPU, embeddings)
    return model


if __name__ == '__main__':
    emb_size = 128
    emb_path = 'data/ast/node_w2v_' + str(emb_size)
    word2vec = Word2Vec.load(emb_path).wv
    # vocab = word2vec.vocab
    embeddings = np.zeros((word2vec.vectors.shape[0] + 1, word2vec.vectors.shape[1]), dtype="float32")
    embeddings[:word2vec.vectors.shape[0]] = word2vec.vectors

    HIDDEN_DIM = 32
    ENCODE_DIM = 128
    USE_GPU = False
    MAX_TOKENS = word2vec.vectors.shape[0]
    EMBEDDING_DIM = word2vec.vectors.shape[1]

    model = BatchProgramEncoder(EMBEDDING_DIM, HIDDEN_DIM, MAX_TOKENS + 1, ENCODE_DIM,
                                   USE_GPU, embeddings)

    json_str = open('data/ast/workload2idx_tree.json').read()
    workload2idx_tree = json.loads(json_str)
    inp = [workload2idx_tree['LassoLars'] for _ in range(2)]

    out = model(inp)
    print(torch.flatten(out))
    print(torch.flatten(out).shape)
