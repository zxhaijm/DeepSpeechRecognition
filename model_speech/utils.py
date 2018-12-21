import os
import numpy as np
import scipy.io.wavfile as wav
from tqdm import tqdm
from scipy.fftpack import fft
from python_speech_features import mfcc


# 对音频文件提取mfcc特征
def compute_mfcc(file):
	fs, audio = wav.read(file)
	mfcc_feat = mfcc(audio, samplerate=fs, numcep=26)
	mfcc_feat = mfcc_feat[::3]
	mfcc_feat = np.transpose(mfcc_feat)
	return mfcc_feat


# 获取信号的时频图
def compute_fbank(file):
	x=np.linspace(0, 400 - 1, 400, dtype = np.int64)
	w = 0.54 - 0.46 * np.cos(2 * np.pi * (x) / (400 - 1) ) # 汉明窗
	fs, wavsignal = wav.read(file)
	# wav波形 加时间窗以及时移10ms
	time_window = 25 # 单位ms
	wav_arr = np.array(wavsignal)
	range0_end = int(len(wavsignal)/fs*1000 - time_window) // 10 # 计算循环终止的位置，也就是最终生成的窗数
	data_input = np.zeros((range0_end, 200), dtype = np.float) # 用于存放最终的频率特征数据
	data_line = np.zeros((1, 400), dtype = np.float)
	for i in range(0, range0_end):
		p_start = i * 160
		p_end = p_start + 400
		data_line = wav_arr[p_start:p_end]	
		data_line = data_line * w # 加窗
		data_line = np.abs(fft(data_line))
		data_input[i]=data_line[0:200] # 设置为400除以2的值（即200）是取一半数据，因为是对称的
	data_input = np.log(data_input + 1)
	#data_input = data_input[::]
	return data_input


class thchs30():
	def __init__(self, datapath='./data/am/wav/data_thchs30'):
		self.datapath = datapath
		self.wav_lst, self.label_lst = self.source_get()
		self.label_data = self.gen_label_data()
		self.vocab = self.mk_vocab()
	
	def source_get(self):
		print('get source list...')
		train_file = self.datapath + '\\data'
		label_lst = []
		wav_lst = []
		for root, _, files in os.walk(train_file):
			for file in files:
				if file.endswith('.wav') or file.endswith('.WAV'):
					wav_file = os.sep.join([root, file])
					label_file = wav_file + '.trn'
					wav_lst.append(wav_file)
					label_lst.append(label_file)
		return wav_lst, label_lst

	def read_label(self, label_file):
		with open(label_file, 'r', encoding='utf8') as f:
			data = f.readlines()
			return data[1]

	def gen_label_data(self):
		print('generate label data...')
		label_data = []
		for label_file in tqdm(self.label_lst):
			pny = self.read_label(label_file)
			label_data.append(pny.strip('\n'))
		return label_data
	
	def mk_vocab(self):
		print('make vocab...')
		vocab = []
		for line in tqdm(self.label_data):
			line = line.split(' ')
			for pny in line:
				if pny not in vocab:
					vocab.append(pny)
		vocab.append('_')
		return vocab
	
	def get_batch(self, batch_size, shuffle_list):
		for i in range(len(self.wav_lst)//batch_size):
			wav_data_lst = []
			label_data_lst = []
			begin = i * batch_size
			end = begin + batch_size
			sub_list = shuffle_list[begin:end]
			for index in sub_list:
				fbank = compute_fbank(self.wav_lst[index])
				pad_fbank = np.zeros((fbank.shape[0]//8*8+8, fbank.shape[1]))
				pad_fbank[:fbank.shape[0], :] = fbank
				label = self.word2id(self.label_data[index])
				wav_data_lst.append(pad_fbank)
				label_data_lst.append(label)
			pad_wav_data, input_length = self.wav_padding(wav_data_lst)
			pad_label_data, label_length = self.label_padding(label_data_lst)
			inputs = {'the_inputs': pad_wav_data,
					'the_labels': pad_label_data,
					'input_length': input_length,
					'label_length': label_length,
					}
			outputs = {'ctc': np.zeros(pad_wav_data.shape[0],)} 
			yield inputs, outputs


	def word2id(self, line):
		return [self.vocab.index(pny) for pny in line.split(' ')]

	def wav_padding(self, wav_data_lst):
		wav_lens = [len(data) for data in wav_data_lst]
		wav_max_len = max(wav_lens)
		wav_lens = np.array([leng//8 for leng in wav_lens])
		new_wav_data_lst = np.zeros((len(wav_data_lst), wav_max_len, 200, 1))
		for i in range(len(wav_data_lst)):
			new_wav_data_lst[i, :wav_data_lst[i].shape[0], :, 0] = wav_data_lst[i]
		return new_wav_data_lst, wav_lens

	def label_padding(self, label_data_lst):
		label_lens = np.array([len(label) for label in label_data_lst])
		max_label_len = max(label_lens)
		new_label_data_lst = np.zeros((len(label_data_lst), max_label_len))
		for i in range(len(label_data_lst)):
			new_label_data_lst[i][:len(label_data_lst[i])] = label_data_lst[i]
		return new_label_data_lst, label_lens
