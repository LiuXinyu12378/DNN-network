"""
项目配置
"""
import pickle
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
############### 排序模型的参数############
dnnsort_max_len = 30
dnn_train_batch_size = 256
dnn_test_batch_size = 500

dnn_ws = pickle.load(open("./models/ws.pkl","rb"))
dnnsort_hidden_size = 128
dnnsort_number_layers = 2
dnnsort_bidriectional = True
dnnsort_lstm_dropout = 0.5


#XX 公司

# 有什么业务
    # 网站，app，外部用户能看到的
    # 公司内部的其他业务
    # -

#重点：
#1. 完成的项目，应该是从简单到难的
    # 1. 数据分析
    # 2. 机器学习
    # 3. 深度学习
#2. 每个项目需要有一个合适的背景

# 数据挖掘，数据分析
    # pandas
    # numpy

# 分类，机器学习等方法试下你的项目

# 文本相似度
    # 召回
    # 排序

# seq2seq
    # 翻译
    #看图说话：RNN+RNN

# 推荐系统 + 机器学习

# 推荐系统 + 深度学习


#项目1
# 1.项目描述：什么时候，做了什么事情，是为了干什么 ，达到了什么效果
# 2.技术点：
    # 2.1 . 使用XXX技术，做什么事情
    # 2.2 . 使用XXX技术，做什么事情
    # 2.3 . 使用XXX技术，做什么事情
    # 2.4 . 使用XXX技术，做什么事情
    # 2.5 . 使用XXX技术，做什么事情
    # 2.6 . 使用XXX技术，做什么事情
    # 2.7 . 使用XXX技术，做什么事情






