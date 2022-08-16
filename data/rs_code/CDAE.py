from .DeepRec_model.cdae import CDAE
import tensorflow as tf
NUM_EPOCHS = 10

def CDAE_application(train_data, test_data, n_user, n_item, ld, lr, bs):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        model = CDAE(sess, n_user, n_item, batch_size=bs, learning_rate=lr, epoch=NUM_EPOCHS)
        model.build_network(hidden_neuron=ld)
        ndcg = model.execute(train_data, test_data)
    return ndcg