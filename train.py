# 1.升学模型训练
import os
import tensorflow as tf
from random import shuffle
from model_speech.cnn_ctc import Am, am_hparams
from model_speech.utils import thchs30

total_num = 100
batch_size = 4
epochs = 1

am_data = thchs30('E:\\DATA\\thchs30\\data_thchs30')

am_args = am_hparams()
am_args.vocab_size = len(am_data.vocab)

am = Am(am_args)

batch_num = total_num // 4
shuffle_list = [i for i in range(total_num)]

if os.path.exists('logs_am/model.h5'):
    print('load acoustic model...')

for k in range(epochs):
    print('this is the', k+1, 'th epochs trainning !!!')
    #shuffle(shuffle_list)
    batch = am_data.get_batch(batch_size, shuffle_list)
    am.ctc_model.fit_generator(batch, steps_per_epoch=batch_num, epochs=1)

am.ctc_model.save_weights('logs_am/model.h5')




# 2.语言模型训练-------------------------------------------
from model_language.transformer import Lm, lm_hparams
from model_language.utils import get_data


total_num = 100
batch_size = 4
epochs = 1

lm_data = get_data("data/lm/lm_data.txt")
lm_args = lm_hparams()
lm_args.input_vocab_size = len(lm_data.pny2id)
lm_args.label_vocab_size = len(lm_data.han2id)

lm = Lm(lm_args)

saver =tf.train.Saver()
with tf.Session() as sess:
    merged = tf.summary.merge_all()
    sess.run(tf.global_variables_initializer())
    if os.path.exists('logs_lm/model.meta'):
        saver.restore(sess, 'logs_lm/model')
    writer = tf.summary.FileWriter('logs_lm/tensorboard', tf.get_default_graph())
    for k in range(epochs):
        total_loss = 0
        batch_num = total_num // batch_size
        batch = lm_data.get_batch(batch_size)
        for i in range(batch_num):
            input_batch, label_batch = next(batch)
            feed = {lm.x: input_batch, lm.y: label_batch}
            cost,_ = sess.run([lm.mean_loss,lm.train_op], feed_dict=feed)
            total_loss += cost
            if (k * batch_num + i) % 10 == 0:
                rs=sess.run(merged, feed_dict=feed)
                writer.add_summary(rs, k * batch_num + i)
        if (k+1) % 5 == 0:
            print('epochs', k+1, ': average loss = ', total_loss/batch_num)
    saver.save(sess, 'logs_lm/model')
    writer.close()