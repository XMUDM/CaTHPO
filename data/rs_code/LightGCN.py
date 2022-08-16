import tensorflow as tf
tf.get_logger().setLevel('ERROR') # only show error messages

from recommenders.utils.timer import Timer
from recommenders.models.deeprec.models.graphrec.lightgcn import LightGCN
from recommenders.models.deeprec.DataModel.ImplicitCF import ImplicitCF
from recommenders.datasets.python_splitters import python_stratified_split
from recommenders.evaluation.python_evaluation import map_at_k, ndcg_at_k, precision_at_k, recall_at_k
from recommenders.utils.constants import SEED as DEFAULT_SEED
from recommenders.models.deeprec.deeprec_utils import prepare_hparams


# top k items to recommend
TOP_K = 10

# Model parameters
EPOCHS = 10

SEED = DEFAULT_SEED  # Set None for non-deterministic results

yaml_file = "./lightgcn.yaml"

def LightGCN_application(data, ld, lr, bs):
    train, test = python_stratified_split(data, ratio=0.75)
    data = ImplicitCF(train=train, test=test, seed=SEED)

    hparams = prepare_hparams(yaml_file,
                              embed_size=ld,
                              n_layers=3,
                              batch_size=bs,
                              epochs=EPOCHS,
                              learning_rate=lr,
                              eval_epoch=5,
                              top_k=TOP_K,
                              )

    model = LightGCN(hparams, data, seed=SEED)

    with Timer() as train_time:
        model.fit()

    topk_scores = model.recommend_k_items(test, top_k=TOP_K, remove_seen=True)

    eval_ndcg = ndcg_at_k(test, topk_scores, k=TOP_K)

    return eval_ndcg
