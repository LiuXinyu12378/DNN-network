"""
实现排序模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import config

class DnnSort(nn.Module):
    def __init__(self):
        super(DnnSort,self).__init__()

        self.embedding = nn.Embedding(num_embeddings=len(config.dnn_ws),
                                      embedding_dim=300,
                                      padding_idx=config.dnn_ws.PAD)
        self.lstm1 = nn.LSTM(
            input_size=300,
            hidden_size=config.dnnsort_hidden_size,
            num_layers=config.dnnsort_number_layers,
            dropout=config.dnnsort_lstm_dropout,
            bidirectional=config.dnnsort_bidriectional,
            batch_first=True
        )

        self.lstm2 = nn.LSTM(
            input_size=config.dnnsort_hidden_size*8,
            hidden_size=config.dnnsort_hidden_size,
            num_layers=config.dnnsort_number_layers,
            dropout=config.dnnsort_lstm_dropout,
            bidirectional=False,
            batch_first=True
        )

        #把编码之后的结果进行DNN，得到 不同类别的概率
        self.fc = nn.Sequential(
            nn.BatchNorm1d(config.dnnsort_hidden_size * 4),

            nn.Linear(config.dnnsort_hidden_size * 4, config.dnnsort_hidden_size * 2),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(config.dnnsort_hidden_size * 2),
            nn.Dropout(config.dnnsort_lstm_dropout),

            nn.Linear(config.dnnsort_hidden_size * 2, config.dnnsort_hidden_size * 2),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(config.dnnsort_hidden_size * 2),
            nn.Dropout(config.dnnsort_lstm_dropout),

            nn.Linear(config.dnnsort_hidden_size * 2, 2),
            # nn.Softmax(dim=-1)
        )

    def forward(self, *input):
        sent1,sent2 = input
        mask1 = sent1.eq(config.dnn_ws.PAD)
        mask2 = sent2.eq(config.dnn_ws.PAD)

        sent1_embeded = self.embedding(sent1)
        sente_embeded = self.embedding(sent2)

        output1,_ = self.lstm1(sent1_embeded)  #[batch_size,seq_len_1,hidden_size*2]
        output2,_ = self.lstm1(sente_embeded)  #[batch_size,seq_len2,hidden_size*2]
        output1_align,output2_align = self.soft_attention_align(output1,output2,mask1,mask2)

        #得到attention result
        #_output1:[batch_size,seq_len1,hidden_size*8]
        #
        _output1 = torch.cat([output1,output1_align,self.submul(output1,output1_align)],dim=-1)
        _output2 = torch.cat([output2,output2_align,self.submul(output2,output2_align)],dim=-1)

        #通过lstm2处理
        output1_composed,_ = self.lstm2(_output1) #[batch_size,seq_len1,hidden_size]
        output2_composed,_ = self.lstm2(_output2) #[batch_size,seq_len2,hidden_size]

        #池化
        output1_pooled = self.apply_pooling(output1_composed)
        output2_pooled = self.apply_pooling(output2_composed)

        x = torch.cat([output1_pooled,output2_pooled],dim=-1) #[bathc-Size,hidden_size*4]
        return self.fc(x)

    def apply_pooling(self,input):
        """
        在seq——len的维度进行池化
        :param input: batch_size,seq_len,hidden_size
        :return:
        """
        avg_output = F.avg_pool1d(input.transpose(1,2),input.size(1)).squeeze(-1) #[batch_size,hidden_size]
        max_output = F.max_pool1d(input.transpose(1,2),input.size(1)).squeeze(-1)
        return torch.cat([avg_output,max_output],dim=-1) #[batch_size,hidden_size*2]


    def submul(self,x1,x2):
        _sub = x1-x2
        _mul = x1*x2
        return torch.cat([_sub,_mul],dim=-1) #[batch_size,seq_len,hidden_size*4]




    def soft_attention_align(self,x1,x2,mask1,mask2):
        """
        进行互相attention，返回context vector
        :param x1: [batch_size,seq_len_1,hidden_size*2]
        :param x2: [batch_size,seq_len_2,hidden_size*2]
        :param mask1: 【batch_size,seq_len1】
        :param mask2: [batch_size,seq_len2】
        :return:
        """
        #0. 把mask替换为-inf
        mask1 = mask1.float().masked_fill_(mask1,float("-inf"))
        mask2 = mask2.float().masked_fill_(mask2,float("-inf"))

        #1. 把一个作为encoder，另一个作为decoder
        #把x1作为encoder，x2作为decoder
        #1. attenion weight:[batch_size,seq_len_2,seq_len_1]
        attention_anergies = torch.bmm(x2,x1.transpose(1,2)) #[batch_size,seq_len2,seq_len1]

        weight1 = F.softmax(attention_anergies+mask1.unsqueeze(1),dim=-1) #[batch_size,seq_len2,seq_len1]

        #2. 得到context vector
        x2_align = torch.bmm(weight1,x1) #[batch_size,seq_len2,hidden_size*2]

        weight2 = F.softmax(attention_anergies.transpose(1,2)+mask2.unsqueeze(1),dim=-1)#[batch_size,seq_len1,seq_len2]
        # 把x2作为encoder，x1作为decoder
        x1_align = torch.bmm(weight2,x2) #[batch_size,seq_len1,hidden_Szie*2]
        return x1_align,x2_align



