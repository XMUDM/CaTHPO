import numpy as np
import pandas as pd
import math

def RMSE(error, num):
    return np.sqrt(error / num)


def MAE(error_mae, num):
    return (error_mae / num)


def NDCG_at_k(df, k):
    if df.shape[1] >= k:
        k = k
    else:
        k = df.shape[1]
    dcg = get_dcg(df["y_pred"], df["y_true"], k)
    idcg = get_dcg(df["y_true"], df["y_true"], k)
    ndcg = dcg / idcg
    return ndcg

def get_dcg(y_pred, y_true, k):
    # 注意y_pred与y_true必须是一一对应的，并且y_pred越大越接近label=1(用相关性的说法就是，与label=1越相关)
    df = pd.DataFrame({"y_pred": y_pred, "y_true": y_true})
    df = df.sort_values(by="y_pred", ascending=False)  # 对y_pred进行降序排列，越排在前面的，越接近label=1
    df = df.iloc[0:k, :]  # 取前K个
    dcg = (2 ** df["y_true"] - 1) / np.log2(np.arange(1, df["y_true"].count() + 1) + 1)  # 位置从1开始计数
    dcg = np.sum(dcg)
    return dcg

def precision_recall_ndcg_at_k(k, rankedlist, test_matrix):
    idcg_k = 0
    dcg_k = 0
    n_k = k if len(test_matrix) > k else len(test_matrix)
    for i in range(n_k):
        idcg_k += 1 / math.log(i + 2, 2)

    b1 = rankedlist
    b2 = test_matrix
    s2 = set(b2)
    hits = [(idx, val) for idx, val in enumerate(b1) if val in s2]
    count = len(hits)

    for c in range(count):
        dcg_k += 1 / math.log(hits[c][0] + 2, 2)

    return float(count / k), float(count / len(test_matrix)), float(dcg_k / idcg_k)


def map_mrr_ndcg(rankedlist, test_matrix):
    ap = 0
    map = 0
    dcg = 0
    idcg = 0
    mrr = 0
    for i in range(len(test_matrix)):
        idcg += 1 / math.log(i + 2, 2)

    b1 = rankedlist
    b2 = test_matrix
    s2 = set(b2)
    hits = [(idx, val) for idx, val in enumerate(b1) if val in s2]
    count = len(hits)

    for c in range(count):
        ap += (c + 1) / (hits[c][0] + 1)
        dcg += 1 / math.log(hits[c][0] + 2, 2)

    if count != 0:
        mrr = 1 / (hits[0][0] + 1)

    if count != 0:
        map = ap / count

    return map, mrr, float(dcg / idcg)


def evaluate(self):
    pred_ratings_10 = {}
    pred_ratings_5 = {}
    pred_ratings = {}
    ranked_list = {}
    p_at_5 = []
    p_at_10 = []
    r_at_5 = []
    r_at_10 = []
    map = []
    mrr = []
    ndcg = []
    ndcg_at_5 = []
    ndcg_at_10 = []
    for u in self.test_users:
        user_ids = []
        user_neg_items = self.neg_items[u]
        item_ids = []
        # scores = []
        for j in user_neg_items:
            item_ids.append(j)
            user_ids.append(u)

        scores = self.predict(user_ids, item_ids)
        # print(type(scores))
        # print(scores)
        # print(np.shape(scores))
        # print(ratings)
        neg_item_index = list(zip(item_ids, scores))

        ranked_list[u] = sorted(neg_item_index, key=lambda tup: tup[1], reverse=True)
        pred_ratings[u] = [r[0] for r in ranked_list[u]]
        pred_ratings_5[u] = pred_ratings[u][:5]
        pred_ratings_10[u] = pred_ratings[u][:10]

        p_5, r_5, ndcg_5 = precision_recall_ndcg_at_k(5, pred_ratings_5[u], self.test_data[u])
        p_at_5.append(p_5)
        r_at_5.append(r_5)
        ndcg_at_5.append(ndcg_5)
        p_10, r_10, ndcg_10 = precision_recall_ndcg_at_k(10, pred_ratings_10[u], self.test_data[u])
        p_at_10.append(p_10)
        r_at_10.append(r_10)
        ndcg_at_10.append(ndcg_10)
        map_u, mrr_u, ndcg_u = map_mrr_ndcg(pred_ratings[u], self.test_data[u])
        map.append(map_u)
        mrr.append(mrr_u)
        ndcg.append(ndcg_u)

    # print("------------------------")
    # print("precision@10:" + str(np.mean(p_at_10)))
    # print("recall@10:" + str(np.mean(r_at_10)))
    # print("precision@5:" + str(np.mean(p_at_5)))
    # print("recall@5:" + str(np.mean(r_at_5)))
    # print("map:" + str(np.mean(map)))
    # print("mrr:" + str(np.mean(mrr)))
    # print("ndcg:" + str(np.mean(ndcg)))
    # print("ndcg@5:" + str(np.mean(ndcg_at_5)))
    # print("ndcg@10:" + str(np.mean(ndcg_at_10)))
    return np.mean(ndcg_at_10)
