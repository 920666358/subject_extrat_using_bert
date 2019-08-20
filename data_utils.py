import numpy as np


def one_hot(y, n_classes):
    y = np.asarray(y, dtype='int32')
    if not n_classes:
        n_classes = np.max(y) + 1
    Y = np.zeros(len(y), n_classes)
    Y[np.arange(len(y)), y] = 1
    return Y


def pad_sequences(sequences, maxlen=None, dtype='int32', padding='post', truncating='post', value=0.):
    """
        sequences: 序列
        maxlen: int 最大长度
        dtype: 转变后的数据类型
        padding: 填充方法'pre' or 'post'
        truncating: 截取方法'pre' or 'post'
        value: float 填充的值
    """
    lengths = [len(s) for s in sequences]
    n_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # 为什么要先初始化为ones，再转换为0’s？可直接初始化zeros 么
    # 如果直接value=0 而不是0.  是不是就不用astype('int32')了？
    # x = np.zeros((n_samples, maxlen)).astype(dtype)
    x = (np.ones((n_samples, maxlen))*value).astype(dtype)

    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError("Truncating type '%s' not understood" % truncating)

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError("padding type '%s' not understood" % padding)
    return x


def shuffle(*arrs, seed=0):
    """*arrs: 数组数据, shuffle数组的第一维"""
    arrs = list(arrs)
    for i, arr in enumerate(arrs):
        assert len(arrs[0]) == len(arrs[i])
        arrs[i] = np.array(arr)
    np.random.seed(seed)
    p = np.random.permutation(len(arrs[0]))  #arrs为数组数据
    return tuple(arr[p] for arr in arrs)

# arrs = [
#     ([1,2,3,4,5],
#     [3,4,5,6,7],
#     [6,7,8,9,10]),
#     ([11,12,13,14,15],
#     [13,14,15,16,17],
#     [16,17,18,19,110]),
# ]
# print(shuffle(arrs))
