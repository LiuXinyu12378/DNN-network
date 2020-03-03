"""
进行模型评估
"""
import torch
import config
from dnnsort import DnnSort
import torch.nn.functional as F
from copy import deepcopy
from cut_sentence import cut


def interface(input1,input2):
    """
    实现预测的逻辑
    :param input1: 用户输入的问题[seq_len]
    :param input2: 召回的问题，[10,seq_len]
    :return:
    """
    input1_list = cut(input1,by_word=True)
    input2_lists = []
    for _input in input2:
        input2_list = cut(_input,by_word=True)
        input2_lists.append(input2_list)
    _input2 = deepcopy(input2)
    input1 = [config.dnn_ws.transform(input1_list, max_len=config.dnnsort_max_len)]*len(input2) #[batch_size,max_len]
    input2 = [config.dnn_ws.transform(i, max_len=config.dnnsort_max_len) for i in input2_lists]  #[batch_size,max_len]
    input1 = torch.LongTensor(input1).to(config.device)
    input2 = torch.LongTensor(input2).to(config.device)
    model = DnnSort().to(config.device)
    model.load_state_dict(torch.load("./models/model.pkl"))
    with torch.no_grad():
        output = model(input1,input2)
        prob = F.softmax(output,dim=-1)
        value,index = prob.max(dim=-1)
        prob_list = []
        for v,i in zip(value,index):
            if i == 0:
                prob_list.append(1- v)  #预测值为0
            else:  #预测值为1
                prob_list.append(v)
    best_ret,best_prob = sorted(zip(_input2,prob_list),key=lambda x:x[-1],reverse=True)[0]
    # if best_prob<0.9:
    #     return "不好意思。。。"
    # else:
    return best_ret


if __name__ == '__main__':
    interface("python好学吗",["python是什么"])
