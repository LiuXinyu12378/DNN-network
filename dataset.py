"""
准备数据集
"""
import random
from tqdm import tqdm
import config
import torch
from torch.utils.data import DataLoader,Dataset
from cut_sentence import cut



#2. 准备dataset

class ChatDataset(Dataset):
    def __init__(self,train=True):
        input_path = "./corpus/q_train.txt" if train else "./corpus/q_test.txt"
        target_path = "./corpus/sim_q_train.txt" if train else  "./corpus/sim_q_test.txt"
        v_path = "./corpus/v_train.txt" if train else  "./corpus/v_test.txt"
        self.input_data = open(input_path,encoding="utf-8").readlines()
        self.target_data = open(target_path,encoding="utf-8").readlines()
        self.v_data = open(v_path,encoding="utf-8").readlines()
        assert len(self.input_data) == len(self.target_data),"input target长度不一致！！！"

    def __getitem__(self, idx):
        input = cut(self.input_data[idx].strip(),by_word=True)
        target = cut(self.target_data[idx].strip(),by_word=True)
        v = int(self.v_data[idx].strip())
        #获取真实长度
        input_len = len(input) if len(input)<config.dnnsort_max_len else config.dnnsort_max_len
        target_len = len(target) if len(target)<config.dnnsort_max_len else config.dnnsort_max_len
        #TODO 修改WS
        input = config.dnn_ws.transform(input,max_len=config.dnnsort_max_len)
        target = config.dnn_ws.transform(target,max_len=config.dnnsort_max_len)
        return input,target,v,input_len,target_len


    def __len__(self):
        return len(self.input_data)



#3. 准备dataloader
def collate_fn(batch):
    """
    :param batch:【（input,target,input_len,target_len），（），（一个getitem的结果）】
    :return:
    """
    #1. 对batch按照input的长度进行排序
    batch = sorted(batch,key=lambda x:x[-2],reverse=True)
    #2. 进行batch操作
    input, target,v, input_len, target_len = zip(*batch)
    #3. 把输入处理成LongTensor
    input = torch.LongTensor(input)
    target = torch.LongTensor(target)
    v = torch.LongTensor(v)  #TODO 作为分类的问题
    input_len = torch.LongTensor(input_len)
    target_len = torch.LongTensor(target_len)
    return input, target,v


def get_dataloader(train=True):
    batch_size = config.dnn_train_batch_size if train else config.dnn_test_batch_size
    return DataLoader(ChatDataset(train),batch_size=batch_size,collate_fn=collate_fn,shuffle=True)