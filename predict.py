from model import Graph
from train import read_data, extract_entity

load_model_path = 'output/subject_model.weights'
train_data, dev_data, test_data, id2class, class2id = read_data()
_, test_model = Graph(0, 0, 0, 0)
test_model.load_weights(load_model_path)


def predict(content, cls):
    return extract_entity(content, cls, class2id, test_model)


if __name__ == '__main__':
    while 1:
        content = input('content: ')
        cls = input('cls: ')
        res = predict(content, cls)
        print(res)
