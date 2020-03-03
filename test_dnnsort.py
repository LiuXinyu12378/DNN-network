"""
测试dnn sort相关的api
"""
from word_sequece import WordSequence
from cut_sentence import cut
import pickle
# import config
from tqdm import tqdm
# from dnnsort.train import train
# from dnnsort.dnnsort import DnnSort
# from torch.optim import Adam
# import torch.nn as nn
# from matplotlib import pyplot as plt

def build_ws():
    ws = WordSequence()
    for line in tqdm(open("corpus/q_train.txt",encoding="utf-8").readlines()):
        ws.fit(cut(line.strip(),by_word=True))
    for line in tqdm(open("corpus/sim_q_train.txt",encoding="utf-8").readlines()):
        ws.fit(cut(line.strip(),by_word=True))

    ws.build_vocab(max_features=None)
    print(len(ws))


    pickle.dump(ws,open("./models/ws.pkl",'wb'))

# def train_dnnsort_model():
#     model = DnnSort().to(config.device)
#     optimizer = Adam(model.parameters())
#     loss_fn = nn.CrossEntropyLoss()
#     # TODO 初始化参数
#     loss_list = []
#     for i in range(10):
#         train(i,model,optimizer,loss_fn,loss_list)
#
#     plt.figure(figsize=(50, 8))
#     plt.plot(range(len(loss_list)), loss_list)
#     plt.show()

if __name__ == '__main__':
    build_ws()
    # train_dnnsort_model()