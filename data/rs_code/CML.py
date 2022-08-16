from .DeepRec_model.cml import CML
import tensorflow as tf
NUM_EPOCHS = 10

def CML_application(train_data, test_data, n_user, n_item, ld, lr, bs):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        model = CML(sess, n_user, n_item, batch_size=bs, learning_rate=lr, epoch=NUM_EPOCHS)
        model.build_network(num_factor=ld)
        ndcg = model.execute(train_data, test_data)
    return ndcg