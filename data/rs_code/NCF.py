import pandas as pd
import tensorflow as tf
tf.get_logger().setLevel('ERROR') # only show error messages
from recommenders.utils.timer import Timer
from recommenders.models.ncf.ncf_singlenode import NCF
from recommenders.models.ncf.dataset import Dataset as NCFDataset
from recommenders.datasets.python_splitters import python_chrono_split
from recommenders.evaluation.python_evaluation import (rmse, mae, rsquared, exp_var, map_at_k, ndcg_at_k, precision_at_k,
                                                     recall_at_k, get_top_k_items)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# top k items to recommend
TOP_K = 10
# Model parameters
EPOCHS = 10
SEED = 42

def NCF_application(data, ld, lr, bs):
    train, test = python_chrono_split(data, 0.75)
    test = test[test["userID"].isin(train["userID"].unique())]
    test = test[test["itemID"].isin(train["itemID"].unique())]
    train_file = "./data/rs_movielen_dataset/ncf_train.csv"
    test_file = "./data/rs_movielen_dataset/ncf_test.csv"
    train.to_csv(train_file, index=False)
    test.to_csv(test_file, index=False)
    data = NCFDataset(train_file=train_file, test_file=test_file, seed=SEED)
    model = NCF (
        n_users=data.n_users,
        n_items=data.n_items,
        model_type="NeuMF",
        n_factors=ld,
        layer_sizes=[16,8,4],
        n_epochs=EPOCHS,
        batch_size=bs,
        learning_rate=lr,
        verbose=20,
        seed=SEED
    )
    with Timer() as train_time:
        model.fit(data)
    with Timer() as test_time:
        users, items, preds = [], [], []
        item = list(train.itemID.unique())
        for user in train.userID.unique():
            user = [user] * len(item)
            users.extend(user)
            items.extend(item)
            preds.extend(list(model.predict(user, item, is_list=True)))

        all_predictions = pd.DataFrame(data={"userID": users, "itemID": items, "prediction": preds})
        merged = pd.merge(train, all_predictions, on=["userID", "itemID"], how="outer")
        all_predictions = merged[merged.rating.isnull()].drop('rating', axis=1)
        eval_ndcg = ndcg_at_k(test, all_predictions, col_prediction='prediction', k=TOP_K)
        return eval_ndcg



