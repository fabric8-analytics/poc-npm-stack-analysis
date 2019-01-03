import numpy as np
import tensorflow as tf
import logging
from aelib.vae import VariationalAutoEncoder
from aelib.utils import *
from scipy import sparse

if __name__ == '__main__':
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True, gpu_options=gpu_options))
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

    np.random.seed(0)
    tf.set_random_seed(0)
    init_logging("vae_pretrain_package_data.log")

    logging.info('loading data')

#     content_matrix_file = open('content_matrix.npy', 'rb')
    # data = np.load(content_matrix_file)
    # content_matrix_file.close()
    csr_sparse = sparse.load_npz('/home/hadoop/CVAE/content_matrix_new.npz')
    # csr_sparse = sparse.load_npz('content_matrix_new.npz')
    data = csr_sparse.toarray()

    idx = np.random.rand(data.shape[0]) < 0.8
    train_X = data[idx]
    test_X = data[~idx]
    logging.info('initializing sdae model')
    model = VariationalAutoEncoder(input_dim=37617, dims=[200, 100], z_dim=50,
                                   activations=['sigmoid', 'sigmoid'], epoch=[50, 50],
                                   noise='mask-0.3', loss='cross-entropy', lr=0.01, batch_size=64, print_step=1)
    logging.info('fitting data starts...')
    model.fit(train_X, test_X)
# feat = model.transform(data)
# scipy.io.savemat('feat-dae.mat',{'feat': feat})
# np.savez("sdae-weights.npz", en_weights=model.weights, en_biases=model.biases,
# 	de_weights=model.de_weights, de_biases=model.de_biases)
