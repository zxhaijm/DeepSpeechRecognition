#coding=utf-8
import os
import tensorflow as tf
from utils import get_data, data_hparams


# 0.准备训练所需数据------------------------------
data_args = data_hparams()
data_args.shuffle = True
data_args.aishell = True
data_args.prime = True
data_args.stcmd = True
data_args.data_length = None
train_data = get_data(data_args)

batch = train_data.get_lm_batch()


while True:
    x, y = next(batch)
