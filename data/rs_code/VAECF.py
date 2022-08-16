import cornac
from recommenders.datasets.python_splitters import python_random_split
from recommenders.evaluation.python_evaluation import map_at_k, ndcg_at_k, precision_at_k, recall_at_k
from recommenders.models.cornac.cornac_utils import predict_ranking
from recommenders.utils.timer import Timer
from recommenders.utils.constants import SEED

TOP_K = 10

def VAECF_application(data, ld, lr, bs):
    train, test = python_random_split(data, 0.75)
    train_set = cornac.data.Dataset.from_uir(train.itertuples(index=False), seed=SEED)
    vaecf = cornac.models.VAECF(
        k=ld,
        autoencoder_structure=[20],
        act_fn="tanh",
        likelihood="mult",
        n_epochs=10,
        batch_size=bs,
        learning_rate=lr,
        beta=1.0,
        seed=123,
        use_gpu=True,
        verbose=True,
    )
    with Timer() as t:
        vaecf.fit(train_set)
    with Timer() as t:
        all_predictions = predict_ranking(vaecf, train, usercol='userID', itemcol='itemID', remove_seen=True)
    eval_ndcg = ndcg_at_k(test, all_predictions, col_prediction='prediction', k=TOP_K)
    return eval_ndcg