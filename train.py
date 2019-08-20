import json
import os
import re
import pandas as pd
import tensorflow as tf
import keras.backend as K
from keras.callbacks import Callback
from keras_bert import Tokenizer, calc_train_steps
from tqdm import tqdm

from model import Graph
from data_utils import *

global graph
graph = tf.get_default_graph()
# set(), 无序不重复元素集
additional_chars = set()
# 假标签法，添加prob>0.9的测试集数据为训练数据 增加模型训练数据量
new_data = []
max_len = 200
category_nums = 21
seed = 10
data_path = 'data/train.csv'

token_dict = {}
dict_path = './chinese_L-12_H-768_A-12/vocab.txt'
with open(dict_path, encoding='utf8') as f:
    for line in f.readlines():
        # 移除首尾空格或换行符
        token = line.strip()
        token_dict[token] = len(token_dict)


class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        tokens = []
        for c in text:
            if c in self._token_dict:
                tokens.append(c)
            elif self._is_space(c):
                tokens.append('[unused1]')
            else:
                tokens.append('[UNK]')
        return tokens


tokenizer = OurTokenizer(token_dict)


def read_data():
    data = pd.read_csv(data_path, header=None)
    data = data[data[2] != '其他']
    data = data[data[1].str.len() <= 256]

    if not os.path.exists('data/classes.json'):
        id2class = dict(enumerate(data[2].unique()))
        class2id = {j: i for i, j in id2class.items()}
        json.dump([id2class, class2id], open('data/train.csv', 'w', encoding='utf8'), ensure_ascii=False)
    else:
        id2class, class2id = json.load(open('data/train.csv', encoding='utf8'))

    train_data = []
    for text, cls, name in zip(data[1], data[2], data[3]):
        if name in text:
            train_data.append((text, cls, name))

    random_order = shuffle(train_data, seed=seed)[0].tolist()
    train_data = random_order[0:int(0.98*len(random_order))]
    val_data = random_order[int(0.98*len(random_order)):]

    new_data = pd.read_csv('data/new_data.csv')
    for text, cls, name in new_data.values:
        train_data.append((text, cls, name))
    train_data = shuffle(train_data, seed=seed)[0].tolist()

    for data_ in train_data+val_data:
        # [^\u4e00-\u9fa5a-zA-Z0-9\*] 中文、英文、数字但不包括下划线等符号
        additional_chars.update(re.findall(u'[^\u4e00-\u9fa5a-zA-Z0-9\*]', data_[2]))
    # 用处?？
    additional_chars.remove(u'，')

    test = pd.read_csv('data/eval.csv', encoding='utf-8', header=None)
    test_data = []
    for id, text, cls in zip(test[0], test[1], test[2]):
        train_data.append((id, text, cls))

    return train_data, val_data, test_data, id2class, class2id


def str_find(str1, str2):
    """在str1中寻找子串str2，找到返回start下标；找不到返回-1"""
    n_str2 = len(str2)
    for i in range(len(str1)):
        if str1[i: i+n_str2] == str2:
            return i
    return -1


def sep_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding]*(ML-len(x))]) if len(x) < ML
        else x
        for x in X
    ])


def data_generator(data, batch_size):
    while 1:
        X, segment, start, end, max_length = [], [], [], [], 0
        for i, d in enumerate(data):
            # 限定text的最大长度为200字符
            text, c = d[0][:max_len], d[1]
            sub = d[2]
            s = text.fine(sub)
            if s != -1:
                e = s+len(sub) - 1
                x, seg = tokenizer.encode(first=c, second=text)

                # each batch,max_length不是固定值么？
                if len(x) > max_length:
                    max_length = len(x)

                X.append(x)
                segment.append(seg)
                start.append(s)
                end.append(e)

                if len(X) == batch_size or i == len(data) -1:
                    X = pad_sequences(X, maxlen=max_length)
                    segment = pad_sequences(segment, maxlen=max_length)
                    start = one_hot(start, max_length)
                    end = one_hot(end, max_length)
                    yield [X, segment, start, end], None
                    X, segment, start, end, max_length = [], [], [], [], 0


def softmax(x):
    x = x-np.max(x)
    x = np.exp(x)
    return x/np.sum(x)


def extract_entity(text, category, class2id, model):
    """解码函数，return公司名称"""
    if category not in class2id.keys():
        return 'NaN'

    text = text[:400]
    x, s = tokenizer.encode(first=category, second=text, max_len=512)
    prob_s, prob_e = model.predict([np.array([x]), np.array([s])])
    prob_s, prob_e = softmax(prob_s[0]), softmax(prob_e[0])

    for i, t in enumerate(text):
        if len(t) == 1 and re.findall(u'[^\u4e00-\u9fa5a-zA-Z0-9\*]', t) and t not in additional_chars:
            prob_s[i] -= 10
    start = prob_s.argmax()

    end = -1
    for e in range(start, len(text)):
        t = text[e]
        if len(t) == 1 and re.findall(u'[^\u4e00-\u9fa5a-zA-Z0-9\*]', t) and t not in additional_chars:
            end = e
            break
    end = prob_e[start: end + 1].argmax() + start

    res = ''.join(text[start: end+1])

    if prob_s[start]>0.9 and prob_e[end] > 0.9:
        new_data.append([text, category, res])

    return res


class Evaluate(Callback):
    def __init__(self, data, model, test_model, class2id):
        self.ACC = []
        self.best = 0.
        self.passed = 0
        self.val_data = data
        self.model = model
        self.test_model = test_model
        self.class2id = class2id

    def evaluate(self):
        eps = 0
        for d in tqdm(iter(self.val_data)):
            R = extract_entity(d[0], d[1], self.class2id, self.test_model)
            if R == d[2]:
                eps += 1
        pre = eps / len(self.val_data)
        # why不可直接返回pre么？
        return 2*pre/(pre+1)

    def on_batch_begin(self, batch, logs=None):
        """第一个epoch用来warmup，第二个epoch把学习率降到最低"""
        if self.passed < self.params['steps']:
            lr = (self.passed + 1.)/self.params['steps']*learning_rate
            K.set_value(self.model.optimizer.lr, lr)
            self.passed += 1
        elif self.params['steps'] <= self.passed < self.params['steps']*2:
            lr = (2-(self.passed+1.)/self.params['steps'])*(learning_rate-min_learning_rate)
            lr += min_learning_rate
            K.set_value(self.model.optimizer.lr, lr)
            self.passed += 1

    def on_batch_end(self, epoch, logs=None):
        acc = self.evaluate()
        self.ACC.append(acc)
        if acc >= self.best:
            self.best = acc
            self.model.save_weights('models/subject_model.weights')
        print('epoch: {}, acc: {}, best acc: {}\n'.format(epoch, acc, self.best))


def val(val_data, class2id, test_model):
    eps = 0
    errs = []
    for d in tqdm(iter(val_data)):
        R = extract_entity(d[0], d[1], class2id, test_model)
        if R == d[2]:
            eps += 1
        else:
            errs.append((d[0], d[1], d[2], R))
    with open('error.txt', 'w', encoding='utf-8') as f:
        f.write(str(errs))

    pre = eps/len(val_data)
    return 2*pre/(pre+1)


def test(test_data, class2id, test_model):
    with open('result.txt', 'w', encoding='utf-8') as f:
        for d in tqdm(iter(test_data)):
            s = str(d[0]+','+extract_entity(d[1].replace('\t', ''), d[2], class2id, test_model))
            f.write(s+'\n')
    import json
    dic = {'data': new_data}
    # dumps是将dict转化成str格式，loads是将str转化成dict格式
    # with open('new_data.txt', 'w', encoding='utf-8') as f:
    #     f.write(json.dumps(dic, ensure_ascii=False))
    json.dump(dic, open('new_data.txt', 'w', encoding='utf-8'), ensure_ascii=False)


if __name__ == '__main__':
    batch_size = 16
    learning_rate = 1e-3
    min_learning_rate = 1e-5
    epochs = 100
    is_test = False

    train_data, val_data, test_data, id2class, class2id = read_data()

    total_steps, warmup_steps = calc_train_steps(
        num_example=len(train_data),
        batch_size=batch_size,
        epochs=epochs,
        warmup_proportion=0.1
    )

    model, test_model = Graph(total_steps, warmup_steps, lr=learning_rate, min_lr=min_learning_rate)

    if is_test:
        test_model.load_weights('models/subject_model.weights')
        model.load_weights('models/subject_model.weights')
        test(test_data, class2id, test_model)
    else:
        evaluator = Evaluate(val_data, model, test_model, class2id)
        X = data_generator(train_data, batch_size)
        steps = int((len(train_data)+batch_size-1)/batch_size)
        model.fit_generator(X,
                            steps_per_epoch=100,
                            epochs=epochs,
                            callbacks=[evaluator])
