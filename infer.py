import numpy as np
from keras import backend as K
from model_speech.cnn_ctc import Am, am_hparams
from model_speech.utils import thchs30

# 1.声学模型解码-------------------------------------------------

def decode_ctc(num_result, num2word):
	result = num_result[:, :, :]
	in_len = np.zeros((1), dtype = np.int32)
	in_len[0] = result.shape[1]
	r = K.ctc_decode(result, in_len, greedy = True, beam_width=10, top_paths=1)
	r1 = K.get_value(r[0][0])
	r1 = r1[0]
	text = []
	for i in r1:
		text.append(num2word[i])
	return r1, text


am_data = thchs30('E:\\DATA\\thchs30\\data_thchs30')

am_args = am_hparams()
am_args.vocab_size = len(am_data.vocab)

am = Am(am_args)
am.ctc_model.load_weights('logs_am/model.h5')

shuffle_list = [i for i in range(100)]
# 测试模型 predict(x, batch_size=None, verbose=0, steps=None)
batch = am_data.get_batch(1, shuffle_list)

for i in range(10):
  # 载入训练好的模型，并进行识别
  inputs, outputs = next(batch)
  x = inputs['the_inputs']
  y = inputs['the_labels'][0]
  result = am.model.predict(x, steps=1)
  # 将数字结果转化为文本结果
  result, text = decode_ctc(result, am_data.vocab)
  print('数字结果： ', result)
  print('文本结果：', text)
  print('原文结果：', [am_data.vocab[int(i)] for i in y])


# 2.语言模型推断-------------------------------------------
import tensorflow as tf
from model_language.transformer import Lm, lm_hparams
from model_language.utils import get_data

lm_data = get_data("data/lm/lm_data.txt")
lm_args = lm_hparams()
lm_args.input_vocab_size = len(lm_data.pny2id)
lm_args.label_vocab_size = len(lm_data.han2id)

lm = Lm(lm_args)

saver =tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, 'logs_lm/model')
    while True:
        line = input('输入测试拼音: ')
        if line == 'exit': break
        line = line.strip('\n').split(' ')
        x = np.array([lm_data.pny2id.index(pny) for pny in line])
        x = x.reshape(1, -1)
        preds = sess.run(lm.preds, {lm.x: x})
        got = ''.join(lm_data.han2id[idx] for idx in preds[0])
        print(got)