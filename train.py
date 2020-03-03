"""
进行模型的训练
"""


import config
import torch
import numpy as np
from dataset import get_dataloader
from tqdm import tqdm
from dnnsort import DnnSort
from torch.optim import Adam
import torch.nn as nn
from matplotlib import pyplot as plt

model = DnnSort().to(config.device)
optimizer = Adam(model.parameters())
loss_fn = nn.CrossEntropyLoss()

loss_list = []
def train(epoch):
    data_loader = get_dataloader(True)
    bar = tqdm(data_loader,total=len(data_loader))
    for idx,(input,target,v) in enumerate(bar):
        input = input.to(config.device)
        target = target.to(config.device)
        v = v.to(config.device)
        optimizer.zero_grad()
        output = model(input,target)
        loss = loss_fn(output,v)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
        bar.set_description("epoch:{}   idx:{}  loss:{:.6f}".format(epoch,idx,np.mean(loss_list)))
        if idx%100 == 0:
            torch.save(model.state_dict(),"./models/model.pkl")
            torch.save(optimizer.state_dict(),"./models/optimizer.pkl")

if __name__ == '__main__':
    for i in range(10):
        train(i)
