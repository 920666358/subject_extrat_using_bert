from keras.layers import *
from keras.models import Model
from keras_bert import load_trained_model_from_checkpoint, AdamWarmup
from keras_layer_normalization import LayerNormalization
import tensorflow as tf
import keras.backend as K

embedding_size = 768
congig_path = './chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'chinese_L-12_H-768_A-12/bert_model.ckpt'

global graph
graph = tf.get_default_graph()


class Attention(Layer):
    pass


def Graph(total_steps, warmup_steps, lr=1e-3, min_lr=1e-5):
    with graph.as_default():
        x_in = Input(shape=(None,))  # 1行none列，2维
        c_in = Input(shape=(None,))
        start_in = Input(shape=(None,))
        end_in = Input(shape=(None,))

        x, c, start, end = x_in, c_in, start_in, end_in
        # (None,,1) 先增加一个维度变成三维的，然后跟0比较得到一个三维的取值为True/False的表，再把True/False转化成浮点数0./1.
        x_mask = (lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(x)
        # 加载模型
        bert = load_trained_model_from_checkpoint(congig_path, checkpoint_path)
        for l in bert.layers:
            l.trainale = True
        # 生成句向量
        x = bert([x, c])
        # start index
        ps1 = Dense(1, use_bias=False)(x)
        # 加mask：将padding的部分置为很小很小的数  1e10=10的10次方
        ps1 = (lambda x: x[0][..., 0] - (1-x[1][..., 0])*1e10)([ps1, x_mask])
        # end index
        ps2 = Dense(1, use_bias=False)(x)
        ps2 = (lambda x: x[0][..., 0] - (1-x[1][..., 0])*1e10)([ps2, x_mask])

        test_model = Model([x_in, c_in], [ps1, ps2])
        train_model = Model([x_in, c_in, start_in, end_in], [ps1, [ps2]])

        loss_1 = K.mean(K.categorical_crossentropy(start_in, ps1, from_logits=True))
        ps2 -= (1-K.cumsum(start, 1))*1e10
        loss_2 = K.mean(K.categorical_crossentropy(end_in, ps2, from_logits=True))
        loss = loss_1 + loss_2

        train_model.add_loss(loss)
        train_model.compile(optimizer=AdamWarmup(total_steps, warmup_steps, min_lr=min_lr, lr=lr))
        train_model.summary()

        return train_model, test_model
