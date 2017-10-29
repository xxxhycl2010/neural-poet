# -*- coding: utf-8 -*-
# Author: XuMing <shibing624@126.com>
# Data: 17/10/29
# Brief: 
config = {
    "train_data_paths": ["./data/cn_poetry.txt"],  # path of training dataset
    "test_data_paths": [""],  # path of testing dataset
    "model_type":"lstm", # model network: lstm, rnn, gru
    "batch_size": 64,  # size of mini-batch
    "num_passes": 50,  # number of passes to run
    "num_layers": 2,  # num of rnn layers
    "dim_nn": 128,  # dim of rnn
    "use_gpu": False,  # use GPU devices
    "num_epoch_to_save_model": 7,  # number of batches to output model
    "save_model_path": "model/poetry.module",  # model path
    "model_path": "model/poetry.module-14",  # model path
}
