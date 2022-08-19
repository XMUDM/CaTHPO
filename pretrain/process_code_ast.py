import os
import numpy as np
import pickle
from tree_sitter import Language, Parser
Language.build_library(
  # Store the library in the `build` directory
  'build/my-languages.so',
  # Include one or more languages
  [
    'tree-sitter-python'
  ]
)
PYTHON_LANGUAGE = Language('build/my-languages.so', 'python')
parser = Parser()
parser.set_language(PYTHON_LANGUAGE)

workloads = ['BiVAE', 'VAECF', 'NCF', 'LightGCN', 'CML', 'CDAE', 'UAutoRec', 'IAutoRec']

def dfs(node, all_nodes):
    all_nodes.append(node)
    node_childs = node.children
    if node_childs == []:
        return
    for i in range(len(node_childs)):
        dfs(node_childs[i], all_nodes)

def build_vocab(vocab_path="vocab"):
    word_count = {}
    for w in workloads:
        file = open("./data/rs_code/" + w + '.py')
        code = file.read()
        words = []
        tree = parser.parse(bytes(code,'utf8'))
        all_nodes = []
        dfs(tree.root_node, all_nodes)
        for item in all_nodes:
            if item.child_count != 0:
                words.append(item.type)
            else:
                words.append(item.text.decode('utf-8'))

        for word in words:
            if word not in word_count:
                word_count[word] = 0
            word_count[word] +=  1

    code_word_count = list(word_count.items())
    code_word_count.sort(key=lambda k: k[1], reverse=True)
    write = open('./pretrain/' + vocab_path + '.code', 'w', encoding='utf-8')
    for word_pair in code_word_count:
       write.write(word_pair[0] + '\t' + str(word_pair[1]) + '\n')
    write.close()


BOS_WORD = '<s>'
EOS_WORD = '</s>'
BLANK_WORD = '<blank>'
UNK_WORD = '<unk>'
def get_w2i():
    code_vocab_file = open('./pretrain/' + 'vocab.code', encoding='utf-8')
    code_w2i = {BLANK_WORD: 0, BOS_WORD: 1, EOS_WORD: 2, UNK_WORD: 3}
    i = 4
    for v in code_vocab_file:
        v = v.split('\t')[0]
        code_w2i[v] = i
        i += 1
    return code_w2i

def fix_length(code, MAX_LEN_CODE=300):
    if len(code) > MAX_LEN_CODE:
        code = code[0:MAX_LEN_CODE]
    else:
        code = np.concatenate((code, [BLANK_WORD for _ in range(MAX_LEN_CODE - len(code))]))
    return code

def word2idx(c, code_w2i):
    code, nl = [], []
    for w in c:
        code.append(code_w2i.get(w, 3))
    return code


def process_code_ast():
    w2i = get_w2i()
    workloas2codeidx = {}
    workload2astidx = {}
    workload2isType = {}
    for w in workloads:
        file = open("./data/rs_code/" + w + ".py")
        code = file.read()
        tree = parser.parse(bytes(code,'utf8'))
        all_nodes = []
        dfs(tree.root_node, all_nodes)

        code_words = []
        ast_words = []
        is_type = []
        for item in all_nodes:
          if item.child_count != 0:
            ast_words.append(item.type)
            is_type.append(True)
          else:
            ast_words.append(item.text.decode('utf-8'))
            code_words.append(item.text.decode('utf-8'))
            is_type.append(False)
        workloas2codeidx[w] = np.array(word2idx(fix_length(code_words), w2i))
        workload2astidx[w] = np.array(word2idx(fix_length(ast_words, 800), w2i))
        workload2isType[w] = is_type
        pickle.dump(workloas2codeidx, open("./pretrain/workload2codeidx.pkl", 'wb'))
        pickle.dump(workload2astidx, open("./pretrain/workload2astidx.pkl", 'wb'))
        pickle.dump(workload2isType, open("./pretrain/workload2isType.pkl", 'wb'))


if __name__ == '__main__':
  build_vocab()
  process_code_ast()
  # print('finish')

