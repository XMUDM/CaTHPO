import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix

from data.rs_code.BiVAE import BiVAE_application
from data.rs_code.LightGCN import LightGCN_application
from data.rs_code.NCF import NCF_application
from data.rs_code.VAECF import VAECF_application
from data.rs_code.IAutoRec import IAutoRec_application
from data.rs_code.UAutoRec import UAutoRec_application
from data.rs_code.CML import CML_application
from data.rs_code.CDAE import CDAE_application

def get_movielens_data_Recommenders(workload, header=["userID", "itemID", "rating"], conf=1, sep="::"):
    if workload in ['NCF', 'LightGCN']:
        header = ["userID", "itemID", "rating", "timestamp"]

    datapath = './data/rs_movielen_dataset/ratings_' + str(conf) + '.dat'
    df = pd.read_csv(
        datapath,
        sep=sep,
        engine="python",
        names=header,
        usecols=[*range(len(header))],
    )
    df[header[2]] = df[header[2]].astype(float)
    return df

def get_movielens_data_DeepRec(workload, conf=1, header=['user_id', 'item_id', 'rating', 't'],
                     test_size=0.25, sep="::"):
    path = './data/rs_movielen_dataset/ratings_' + str(conf) + '.dat'
    df = pd.read_csv(
        path,
        sep=sep,
        engine="python",
        names=header,
        usecols=[*range(len(header))],
    )
    n_users = df.user_id.unique().shape[0]
    n_items = df.item_id.unique().shape[0]

    train_data, test_data = train_test_split(df, test_size=test_size)
    train_data = pd.DataFrame(train_data)
    test_data = pd.DataFrame(test_data)

    train_row = []
    train_col = []
    train_rating = []

    for line in train_data.itertuples():
        u = line[1] - 1
        i = line[2] - 1
        train_row.append(u)
        train_col.append(i)
        train_rating.append(line[3])
    train_matrix = csr_matrix((train_rating, (train_row, train_col)), shape=(n_users, n_items))

    test_row = []
    test_col = []
    test_rating = []
    for line in test_data.itertuples():
        test_row.append(line[1] - 1)
        test_col.append(line[2] - 1)
        test_rating.append(line[3])
    test_matrix = csr_matrix((test_rating, (test_row, test_col)), shape=(n_users, n_items))
    print("Load data finished. Number of users:", n_users, "Number of items:", n_items)
    return train_matrix.todok(), test_matrix.todok(), n_users, n_items

def get_movielens_data_neg(workload, conf=1, header=['user_id', 'item_id', 'rating', 't'],
                     test_size=0.25, sep="::"):
    path = './data/rs_movielen_dataset/ratings_' + str(conf) + '.dat'
    df = pd.read_csv(
        path,
        sep=sep,
        engine="python",
        names=header,
        usecols=[*range(len(header))],
    )

    n_users = df.user_id.unique().shape[0]
    n_items = df.item_id.unique().shape[0]

    train_data, test_data = train_test_split(df, test_size=test_size)
    train_data = pd.DataFrame(train_data)
    test_data = pd.DataFrame(test_data)

    train_row = []
    train_col = []
    train_rating = []

    for line in train_data.itertuples():
        u = line[1] - 1
        i = line[2] - 1
        train_row.append(u)
        train_col.append(i)
        train_rating.append(1)
    train_matrix = csr_matrix((train_rating, (train_row, train_col)), shape=(n_users, n_items))

    test_row = []
    test_col = []
    test_rating = []
    for line in test_data.itertuples():
        test_row.append(line[1] - 1)
        test_col.append(line[2] - 1)
        test_rating.append(1)
    test_matrix = csr_matrix((test_rating, (test_row, test_col)), shape=(n_users, n_items))

    test_dict = {}
    for u in range(n_users):
        test_dict[u] = test_matrix.getrow(u).nonzero()[1]

    print("Load data finished. Number of users:", n_users, "Number of items:", n_items)
    return train_matrix.todok(), test_dict, n_users, n_items

def get_movielens_data_cdae(workload, conf=1, header=['user_id', 'item_id', 'rating', 't'],
                     test_size=0.25, sep="::"):
    path = './data/rs_movielen_dataset/ratings_' + str(conf) + '.dat'
    df = pd.read_csv(path, sep=sep, names=header, engine='python')

    n_users = df.user_id.unique().shape[0]
    n_items = df.item_id.unique().shape[0]

    train_data, test_data = train_test_split(df, test_size=test_size)
    train_data = pd.DataFrame(train_data)
    test_data = pd.DataFrame(test_data)

    train_row = []
    train_col = []
    train_rating = []

    train_dict = {}
    for line in train_data.itertuples():
        u = line[1] - 1
        i = line[2] - 1
        train_dict[(u, i)] = 1

    for u in range(n_users):
        for i in range(n_items):
            train_row.append(u)
            train_col.append(i)
            if (u, i) in train_dict.keys():
                train_rating.append(1)
            else:
                train_rating.append(0)
    train_matrix = csr_matrix((train_rating, (train_row, train_col)), shape=(n_users, n_items))
    all_items = set(np.arange(n_items))

    neg_items = {}
    train_interaction_matrix = []
    for u in range(n_users):
        neg_items[u] = list(all_items - set(train_matrix.getrow(u).nonzero()[1]))
        train_interaction_matrix.append(list(train_matrix.getrow(u).toarray()[0]))

    test_row = []
    test_col = []
    test_rating = []
    for line in test_data.itertuples():
        test_row.append(line[1] - 1)
        test_col.append(line[2] - 1)
        test_rating.append(1)
    test_matrix = csr_matrix((test_rating, (test_row, test_col)), shape=(n_users, n_items))

    test_dict = {}
    for u in range(n_users):
        test_dict[u] = test_matrix.getrow(u).nonzero()[1]

    print("Load data finished. Number of users:", n_users, "Number of items:", n_items)

    return train_interaction_matrix, test_dict, n_users, n_items

def run_bench(workload, dataset, param):
    if dataset < 1 or dataset > 5:
        code = 1
        msg = 'error: no dataset'
        y = -1
        return code, msg, y

    ld, lr, bs = param
    lr /= 10000
    y = -1
    if workload in ['BiVAE', 'VAECF', 'NCF', 'LightGCN']:
        data = get_movielens_data_Recommenders(workload, conf=dataset)
        if workload == 'BiVAE':
            y = BiVAE_application(data, ld, lr, bs)
        elif workload == 'VAECF':
            y = VAECF_application(data, ld, lr, bs)
        elif workload == 'NCF':
            y = NCF_application(data, ld, lr, bs)
        elif workload == 'LightGCN':
            y = LightGCN_application(data, ld, lr, bs)
    elif workload in ['CML']:
        train_data, test_data, n_user, n_item = get_movielens_data_neg(workload=workload, conf=dataset)
        if workload == 'CML':
            y = CML_application(train_data, test_data, n_user, n_item, ld, lr, bs)
    elif workload in ['CDAE']:
        train_data, test_data, n_user, n_item = get_movielens_data_cdae(workload=workload, conf=dataset)
        if workload == 'CDAE':
            y = CDAE_application(train_data, test_data, n_user, n_item, ld, lr, bs)
    else:
        train_data, test_data, n_user, n_item = get_movielens_data_DeepRec(workload=workload, conf=dataset)
        if workload == 'MF':
            y = MF_application(train_data, test_data, n_user, n_item, ld, lr, bs)
        elif workload == 'UAutoRec':
            y = UAutoRec_application(train_data, test_data, n_user, n_item, ld, lr, bs)
        elif workload == 'IAutoRec':
            y = IAutoRec_application(train_data, test_data, n_user, n_item, ld, lr, bs)
        else:
            code = 1
            msg = 'error: no workload'
            y = 0
            return code, msg, y
        
    y *= 100
    if y>=0 and y<=100:
        code = 0
        msg = 'run ' + workload + ' exp success! NDCG: ' + str(y)
    else:
        code = 1
        y = -1
        msg = 'run ' + workload + ' exp error!'

    return code, msg, y





