import torch
import cornac
from recommenders.datasets.python_splitters import python_random_split
from recommenders.evaluation.python_evaluation import map_at_k, ndcg_at_k, precision_at_k, recall_at_k
from recommenders.models.cornac.cornac_utils import predict_ranking
from recommenders.utils.timer import Timer
from recommenders.utils.constants import SEED

TOP_K = 10
ENCODER_DIMS = [100]
ACT_FUNC = "tanh"
LIKELIHOOD = "pois"
NUM_EPOCHS = 50

def BiVAE_application(data, ld, lr, bs):
    train, test = python_random_split(data, 0.75)
    train_set = cornac.data.Dataset.from_uir(train.itertuples(index=False), seed=SEED)
    bivae = cornac.models.BiVAECF(
        k=ld,
        encoder_structure=ENCODER_DIMS,
        act_fn=ACT_FUNC,
        likelihood=LIKELIHOOD,
        n_epochs=NUM_EPOCHS,
        batch_size=bs,
        learning_rate=lr,
        seed=SEED,
        use_gpu=torch.cuda.is_available(),
        verbose=True
    )

    with Timer() as t:
        bivae.fit(train_set)

    with Timer() as t:
        all_predictions = predict_ranking(bivae, train, usercol='userID', itemcol='itemID', remove_seen=True)

    eval_ndcg = ndcg_at_k(test, all_predictions, col_prediction='prediction', k=TOP_K)

    return eval_ndcg
