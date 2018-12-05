from easydict import EasyDict as edict


def get_config():
    config = edict()

    config.cuda_id = "cuda:3"
    config.data_dir = './data/test_data/'

    config.margin = 1.0
    config.train_n_classes = 2
    config.test_n_classes = 2
    config.train_n_samples = 3
    config.test_n_samples = 3

    config.lr = 1e-3
    config.weight_decay = 1e-4
    config.step_size = 8
    config.gamma = 0.1
    config.last_epoch = -1
    config.n_epochs = 5
    config.log_interval = 5

    return config

