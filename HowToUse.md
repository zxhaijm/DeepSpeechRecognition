
# 1. 声学模型训练

train.py文件


```python
import os
import tensorflow as tf
from utils import get_data, data_hparams


# 准备训练所需数据
data_args = data_hparams()
data_args.data_length = 10
train_data = get_data(data_args)


# 1.声学模型训练-----------------------------------
from model_speech.cnn_ctc import Am, am_hparams
am_args = am_hparams()
am_args.vocab_size = len(train_data.am_vocab)
am = Am(am_args)
if os.path.exists('logs_am/model.h5'):
    print('load acoustic model...')
    am.ctc_model.load_weights('logs_am/model.h5')

epochs = 0
batch_num = len(train_data.wav_lst) // train_data.batch_size

for k in range(epochs):
    print('this is the', k+1, 'th epochs trainning !!!')
    #shuffle(shuffle_list)
    batch = train_data.get_am_batch()
    am.ctc_model.fit_generator(batch, steps_per_epoch=batch_num, epochs=1)

am.ctc_model.save_weights('logs_am/model.h5')

```

    get source list...
    load  thchs_train.txt  data...
    

    100%|████████████████████████████████████████████████████████████████████████| 10000/10000 [00:00<00:00, 236865.96it/s]
    

    load  aishell_train.txt  data...
    

    100%|██████████████████████████████████████████████████████████████████████| 120098/120098 [00:00<00:00, 260863.15it/s]
    

    make am vocab...
    

    100%|████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 9986.44it/s]
    

    make lm pinyin vocab...
    

    100%|████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 9946.18it/s]
    

    make lm hanzi vocab...
    

    100%|████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 9950.90it/s]
    Using TensorFlow backend.
    

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    the_inputs (InputLayer)      (None, None, 200, 1)      0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, None, 200, 32)     320       
    _________________________________________________________________
    batch_normalization_1 (Batch (None, None, 200, 32)     128       
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, None, 200, 32)     9248      
    _________________________________________________________________
    batch_normalization_2 (Batch (None, None, 200, 32)     128       
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, None, 100, 32)     0         
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, None, 100, 64)     18496     
    _________________________________________________________________
    batch_normalization_3 (Batch (None, None, 100, 64)     256       
    _________________________________________________________________
    conv2d_4 (Conv2D)            (None, None, 100, 64)     36928     
    _________________________________________________________________
    batch_normalization_4 (Batch (None, None, 100, 64)     256       
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, None, 50, 64)      0         
    _________________________________________________________________
    conv2d_5 (Conv2D)            (None, None, 50, 128)     73856     
    _________________________________________________________________
    batch_normalization_5 (Batch (None, None, 50, 128)     512       
    _________________________________________________________________
    conv2d_6 (Conv2D)            (None, None, 50, 128)     147584    
    _________________________________________________________________
    batch_normalization_6 (Batch (None, None, 50, 128)     512       
    _________________________________________________________________
    max_pooling2d_3 (MaxPooling2 (None, None, 25, 128)     0         
    _________________________________________________________________
    conv2d_7 (Conv2D)            (None, None, 25, 128)     147584    
    _________________________________________________________________
    batch_normalization_7 (Batch (None, None, 25, 128)     512       
    _________________________________________________________________
    conv2d_8 (Conv2D)            (None, None, 25, 128)     147584    
    _________________________________________________________________
    batch_normalization_8 (Batch (None, None, 25, 128)     512       
    _________________________________________________________________
    conv2d_9 (Conv2D)            (None, None, 25, 128)     147584    
    _________________________________________________________________
    batch_normalization_9 (Batch (None, None, 25, 128)     512       
    _________________________________________________________________
    conv2d_10 (Conv2D)           (None, None, 25, 128)     147584    
    _________________________________________________________________
    batch_normalization_10 (Batc (None, None, 25, 128)     512       
    _________________________________________________________________
    reshape_1 (Reshape)          (None, None, 3200)        0         
    _________________________________________________________________
    dense_1 (Dense)              (None, None, 256)         819456    
    _________________________________________________________________
    dense_2 (Dense)              (None, None, 230)         59110     
    =================================================================
    Total params: 1,759,174
    Trainable params: 1,757,254
    Non-trainable params: 1,920
    _________________________________________________________________
    load acoustic model...
    

# 2.语言模型训练


```python
# 2.语言模型训练-------------------------------------------
from model_language.transformer import Lm, lm_hparams


lm_args = lm_hparams()
lm_args.input_vocab_size = len(train_data.pny_vocab)
lm_args.label_vocab_size = len(train_data.han_vocab)
lm = Lm(lm_args)

epochs = 0
with lm.graph.as_default():
    saver =tf.train.Saver()
with tf.Session(graph=lm.graph) as sess:
    merged = tf.summary.merge_all()
    sess.run(tf.global_variables_initializer())
    if os.path.exists('logs_lm/model.meta'):
        print('loading language model...')
        saver.restore(sess, 'logs_lm/model')
    writer = tf.summary.FileWriter('logs_lm/tensorboard', tf.get_default_graph())
    for k in range(epochs):
        total_loss = 0
        batch = train_data.get_lm_batch()
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
```

    loading language model...
    INFO:tensorflow:Restoring parameters from logs_lm/model
    

# 3. 模型测试
整合声学模型和语言模型

test.py文件


### 定义解码器


```python
import os
import tensorflow as tf
import numpy as np
from keras import backend as K

# 定义解码器------------------------------------
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
```

### 准备测试数据


```python
# 0. 准备解码所需字典，需和训练一致，也可以将字典保存到本地，直接进行读取
from utils import get_data, data_hparams
data_args = data_hparams()
data_args.data_length = 10 # 重新训练需要注释该行
train_data = get_data(data_args)


# 3. 准备测试所需数据， 不必和训练数据一致，通过设置data_args.data_type测试，
#    此处应设为'test'，我用了'train'因为演示模型较小，如果使用'test'看不出效果，
#    且会出现未出现的词。
data_args.data_type = 'train'
test_data = get_data(data_args)
am_batch = train_data.get_am_batch()
lm_batch = train_data.get_lm_batch()
```

    get source list...
    load  thchs_train.txt  data...
    

    100%|████████████████████████████████████████████████████████████████████████| 10000/10000 [00:00<00:00, 226097.06it/s]
    

    load  aishell_train.txt  data...
    

    100%|██████████████████████████████████████████████████████████████████████| 120098/120098 [00:00<00:00, 226827.96it/s]
    

    make am vocab...
    

    100%|████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 9950.90it/s]
    

    make lm pinyin vocab...
    

    100%|██████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<?, ?it/s]
    

    make lm hanzi vocab...
    

    100%|████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 9953.26it/s]
    

### 加载声学模型和语言模型


```python
# 1.声学模型-----------------------------------
from model_speech.cnn_ctc import Am, am_hparams

am_args = am_hparams()
am_args.vocab_size = len(train_data.am_vocab)
am = Am(am_args)
print('loading acoustic model...')
am.ctc_model.load_weights('logs_am/model.h5')

# 2.语言模型-------------------------------------------
from model_language.transformer import Lm, lm_hparams

lm_args = lm_hparams()
lm_args.input_vocab_size = len(train_data.pny_vocab)
lm_args.label_vocab_size = len(train_data.han_vocab)
print('loading language model...')
lm = Lm(lm_args)
sess = tf.Session(graph=lm.graph)
with lm.graph.as_default():
    saver =tf.train.Saver()
with sess.as_default():
    saver.restore(sess, 'logs_lm/model')
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    the_inputs (InputLayer)      (None, None, 200, 1)      0         
    _________________________________________________________________
    conv2d_11 (Conv2D)           (None, None, 200, 32)     320       
    _________________________________________________________________
    batch_normalization_11 (Batc (None, None, 200, 32)     128       
    _________________________________________________________________
    conv2d_12 (Conv2D)           (None, None, 200, 32)     9248      
    _________________________________________________________________
    batch_normalization_12 (Batc (None, None, 200, 32)     128       
    _________________________________________________________________
    max_pooling2d_4 (MaxPooling2 (None, None, 100, 32)     0         
    _________________________________________________________________
    conv2d_13 (Conv2D)           (None, None, 100, 64)     18496     
    _________________________________________________________________
    batch_normalization_13 (Batc (None, None, 100, 64)     256       
    _________________________________________________________________
    conv2d_14 (Conv2D)           (None, None, 100, 64)     36928     
    _________________________________________________________________
    batch_normalization_14 (Batc (None, None, 100, 64)     256       
    _________________________________________________________________
    max_pooling2d_5 (MaxPooling2 (None, None, 50, 64)      0         
    _________________________________________________________________
    conv2d_15 (Conv2D)           (None, None, 50, 128)     73856     
    _________________________________________________________________
    batch_normalization_15 (Batc (None, None, 50, 128)     512       
    _________________________________________________________________
    conv2d_16 (Conv2D)           (None, None, 50, 128)     147584    
    _________________________________________________________________
    batch_normalization_16 (Batc (None, None, 50, 128)     512       
    _________________________________________________________________
    max_pooling2d_6 (MaxPooling2 (None, None, 25, 128)     0         
    _________________________________________________________________
    conv2d_17 (Conv2D)           (None, None, 25, 128)     147584    
    _________________________________________________________________
    batch_normalization_17 (Batc (None, None, 25, 128)     512       
    _________________________________________________________________
    conv2d_18 (Conv2D)           (None, None, 25, 128)     147584    
    _________________________________________________________________
    batch_normalization_18 (Batc (None, None, 25, 128)     512       
    _________________________________________________________________
    conv2d_19 (Conv2D)           (None, None, 25, 128)     147584    
    _________________________________________________________________
    batch_normalization_19 (Batc (None, None, 25, 128)     512       
    _________________________________________________________________
    conv2d_20 (Conv2D)           (None, None, 25, 128)     147584    
    _________________________________________________________________
    batch_normalization_20 (Batc (None, None, 25, 128)     512       
    _________________________________________________________________
    reshape_2 (Reshape)          (None, None, 3200)        0         
    _________________________________________________________________
    dense_3 (Dense)              (None, None, 256)         819456    
    _________________________________________________________________
    dense_4 (Dense)              (None, None, 230)         59110     
    =================================================================
    Total params: 1,759,174
    Trainable params: 1,757,254
    Non-trainable params: 1,920
    _________________________________________________________________
    loading acoustic model...
    loading language model...
    INFO:tensorflow:Restoring parameters from logs_lm/model
    

### 使用语音识别系统


```python

for i in range(5):
    print('\n the ', i, 'th example.')
    # 载入训练好的模型，并进行识别
    inputs, outputs = next(am_batch)
    x = inputs['the_inputs']
    y = inputs['the_labels'][0]
    result = am.model.predict(x, steps=1)
    # 将数字结果转化为文本结果
    _, text = decode_ctc(result, train_data.am_vocab)
    text = ' '.join(text)
    print('文本结果：', text)
    print('原文结果：', ' '.join([train_data.am_vocab[int(i)] for i in y]))
    with sess.as_default():
        _, y = next(lm_batch)
        text = text.strip('\n').split(' ')
        x = np.array([train_data.pny_vocab.index(pny) for pny in text])
        x = x.reshape(1, -1)
        preds = sess.run(lm.preds, {lm.x: x})
        got = ''.join(train_data.han_vocab[idx] for idx in preds[0])
        print('原文汉字：', ''.join(train_data.han_vocab[idx] for idx in y[0]))
        print('识别结果：', got)
sess.close()
```

    
     the  0 th example.
    WARNING:tensorflow:From c:\users\administrator\appdata\local\programs\python\python36\lib\site-packages\keras\backend\tensorflow_backend.py:4303: sparse_to_dense (from tensorflow.python.ops.sparse_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Create a `tf.sparse.SparseTensor` and use `tf.sparse.to_dense` instead.
    文本结果： lv4 shi4 yang2 chun1 yan1 jing3 da4 kuai4 wen2 zhang1 de di3 se4 si4 yue4 de lin2 luan2 geng4 shi4 lv4 de2 xian1 huo2 xiu4 mei4 shi1 yi4 ang4 ran2
    原文结果： lv4 shi4 yang2 chun1 yan1 jing3 da4 kuai4 wen2 zhang1 de di3 se4 si4 yue4 de lin2 luan2 geng4 shi4 lv4 de2 xian1 huo2 xiu4 mei4 shi1 yi4 ang4 ran2
    原文汉字： 绿是阳春烟景大块文章的底色四月的林峦更是绿得鲜活秀媚诗意盎然
    识别结果： 绿是阳春烟景大块文章的底色四月的林峦更是绿得鲜活秀媚诗意盎然
    
     the  1 th example.
    文本结果： ta1 jin3 ping2 yao1 bu4 de li4 liang4 zai4 yong3 dao4 shang4 xia4 fan1 teng2 yong3 dong4 she2 xing2 zhuang4 ru2 hai3 tun2 yi4 zhi2 yi3 yi1 tou2 de you1 shi4 ling3 xian1
    原文结果： ta1 jin3 ping2 yao1 bu4 de li4 liang4 zai4 yong3 dao4 shang4 xia4 fan1 teng2 yong3 dong4 she2 xing2 zhuang4 ru2 hai3 tun2 yi4 zhi2 yi3 yi1 tou2 de you1 shi4 ling3 xian1
    原文汉字： 他仅凭腰部的力量在泳道上下翻腾蛹动蛇行状如海豚一直以一头的优势领先
    识别结果： 他仅凭腰部的力量在泳道上下翻腾蛹动蛇行状如海豚一直以一头的优势领先
    
     the  2 th example.
    文本结果： pao4 yan3 da3 hao3 le zha4 yao4 zen3 me zhuang1 yue4 zheng4 cai2 yao3 le yao3 ya2 shu1 di4 tuo1 qu4 yi1 fu2 guang1 bang3 zi chong1 jin4 le shui3 cuan4 dong4
    原文结果： pao4 yan3 da3 hao3 le zha4 yao4 zen3 me zhuang1 yue4 zheng4 cai2 yao3 le yao3 ya2 shu1 di4 tuo1 qu4 yi1 fu2 guang1 bang3 zi chong1 jin4 le shui3 cuan4 dong4
    原文汉字： 炮眼打好了炸药怎么装岳正才咬了咬牙倏地脱去衣服光膀子冲进了水窜洞
    识别结果： 炮眼打好了炸药怎么装岳正才咬了咬牙倏地脱去衣服光膀子冲进了水窜洞
    
     the  3 th example.
    文本结果： ke3 shei2 zhi1 wen2 wan2 hou4 ta1 yi1 zhao4 jing4 zi zhi1 jian4 zuo3 xia4 yan3 jian3 de xian4 you4 cu1 you4 hei1 yu3 you4 ce4 ming2 xian3 bu2 dui4 cheng1
    原文结果： ke3 shei2 zhi1 wen2 wan2 hou4 ta1 yi1 zhao4 jing4 zi zhi1 jian4 zuo3 xia4 yan3 jian3 de xian4 you4 cu1 you4 hei1 yu3 you4 ce4 ming2 xian3 bu2 dui4 cheng1
    原文汉字： 可谁知纹完后她一照镜子只见左下眼睑的线又粗又黑与右侧明显不对称
    识别结果： 可谁知纹完后她一照镜子知见左下眼睑的线右粗右黑与右侧明显不对称
    
     the  4 th example.
    文本结果： yi1 jin4 men2 wo3 bei4 jing1 dai1 le zhe4 hu4 ming2 jiao4 pang2 ji2 de lao3 nong2 shi4 kang4 mei3 yuan2 chao2 fu4 shang1 hui2 xiang1 de lao3 bing1 qi1 zi3 chang2 nian2 you3 bing4 jia1 tu2 si4 bi4 yi1 pin2 ru2 xi3
    原文结果： yi1 jin4 men2 wo3 bei4 jing1 dai1 le zhe4 hu4 ming2 jiao4 pang2 ji2 de lao3 nong2 shi4 kang4 mei3 yuan2 chao2 fu4 shang1 hui2 xiang1 de lao3 bing1 qi1 zi3 chang2 nian2 you3 bing4 jia1 tu2 si4 bi4 yi1 pin2 ru2 xi3
    原文汉字： 一进门我被惊呆了这户名叫庞吉的老农是抗美援朝负伤回乡的老兵妻子长年有病家徒四壁一贫如洗
    识别结果： 一进门我被惊呆了这户名叫庞吉的老农是抗美援朝负伤回乡的老兵妻子长年有病家徒四壁一贫如洗
    


```python

```
