from tqdm import tqdm
import numpy as np


class get_data():
    def __init__(self, filepath="data/lm/lm_data.txt"):
        print('open data source...')
        with open(filepath, 'r', encoding='utf-8') as f:
            self.data = f.readlines()[:100]
        print('seprate data into inputs and labels...')
        self.inputs, self.labels = self.get_data()
        print('make vocab...')
        self.pny2id = self.get_vocab(self.inputs)
        self.han2id = self.get_vocab(self.labels)
        print('transform symbol to index...')
        self.input_num, self.label_num = self.symbol2id()

    def get_data(self):
        inputs = []
        labels = []
        for i in tqdm(range(len(self.data))):
            _, pny, hanzi = self.data[i].split('\t')
            inputs.append(pny.split(' '))
            labels.append(hanzi.strip('\n').split(' '))
        return inputs, labels

    def get_vocab(self, data):
        vocab = ['<PAD>']
        for line in tqdm(data):
            for char in line:
                if char not in vocab:
                    vocab.append(char)
        return vocab

    def symbol2id(self):
        input_num = [[self.pny2id.index(pny) for pny in line] for line in tqdm(self.inputs)]
        label_num = [[self.han2id.index(han) for han in line] for line in tqdm(self.labels)]
        return input_num, label_num
    
    def get_batch(self, batch_size):
        batch_num = len(self.input_num) // batch_size
        for k in range(batch_num):
            begin = k * batch_size
            end = begin + batch_size
            input_batch = self.input_num[begin:end]
            label_batch = self.label_num[begin:end]
            max_len = max([len(line) for line in input_batch])
            input_batch = np.array([line + [0] * (max_len - len(line)) for line in input_batch])
            label_batch = np.array([line + [0] * (max_len - len(line)) for line in label_batch])
            yield input_batch, label_batch
