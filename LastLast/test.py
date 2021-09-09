import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional
from torch.nn.init import xavier_normal_
import torch.nn.functional as F


class Model(nn.Module):
    ####################################################################
    # v=resnet152-------v
    #              MutanFusion---(注意力机制)---->y    softmax(y)==out
    # q=GRU ------------^
    # Mutan主要将V，q融合。
    ####################################################################
    def __init__(self, d, tv, tq, to, layers=[64, 32, 16, 8]):
        super(Model, self).__init__()
        #  秩约束mutanfusion
        self.R = 10
        self.linear_v = nn.Linear(128, 8)  # 2048-->310
        self.linear_q = nn.Linear(128, 8)  # 2400-->310
        self.list_linear_hv = nn.ModuleList([
            nn.Linear(8, 32)  # 310-->510
            for i in range(self.R)])  # R = 5
        self.list_linear_hq = nn.ModuleList([
            nn.Linear(8, 32)  # Linear(310, 510)
            for i in range(self.R)])  # 5

        # MLP
        self.MLP_Embedding_User = torch.nn.Embedding(200, layers[0])
        self.Layer1 = torch.nn.Linear(96, layers[1])
        self.Layer3 = torch.nn.Linear(layers[1], layers[2])
        self.Layer4 = torch.nn.Linear(layers[2], layers[3])
        self.Layer5 = torch.nn.Linear(layers[3], 1)
        self.dropout_v = 0.5
        self.dropout_q = 0.5
        self.dropout_hv = 0
        self.dropout_hq = 0
        self.loss = torch.nn.BCELoss()

    def init(self):
        xavier_normal_(self.MLP_Embedding_User.weight.data)

    def forward(self, input_v, input_q, user_input):
        # input_v和input_q的维度都是2, (batchsize, d_v) ,统一输入图像和问题的维度
        batch_size = input_v.size(0)

        # 分别处理，图像和问题嵌入
        x_v = F.dropout(input_v, p=self.dropout_v, training=self.training)  # dropout_v = 0.5
        x_v = self.linear_v(x_v)  # (128, 4)
        x_v = getattr(F, 'tanh')(x_v)  # tanh
        x_q = F.dropout(input_q, p=self.dropout_q, training=self.training)
        x_q = self.linear_q(x_q)
        x_q = getattr(F, 'tanh')(x_q)

        # 秩R的约束，（论文中）Z表示成R个Zr的总和（Z会投影到预测空间y上）。
        # 处理后的图像和问题，使用了对应位的相乘，
        # 使用堆叠求和方式进行相加，最终得到的x_mm相当于文章的Z
        x_mm = []
        for i in range(self.R):  # R个映射独立的进行映射，存储到x_mm  R = 5
            x_hv = F.dropout(x_v, p=self.dropout_hv, training=self.training)  #  0
            x_hv = self.list_linear_hv[i](x_hv)  # linear后大小变32， (4, 32)
            x_hv = getattr(F, 'tanh')(x_hv)
            x_hq = F.dropout(x_q, p=self.dropout_hq, training=self.training)
            x_hq = self.list_linear_hq[i](x_hq)
            x_hq = getattr(F, 'tanh')(x_hq)
            x_mm.append(torch.mul(x_hq, x_hv))  # 使用mul（）对应位相乘进行融合，这样融合之后大小不变，但是有R个

        x_mm = torch.stack(x_mm, dim=1)  # R个，，在维度1堆起来，
        item_inputs = x_mm.sum(1).view(batch_size, 32)  # dim1求和，恢复原来大小（batchsize,32）

        # item_inputs = getattr(F, 'softmax')(item_inputs)  # activation_mm = softmax
        user = self.MLP_Embedding_User(user_input)
        user_inputs = user.view(user_input.size(0), -1)
        vector = torch.cat([item_inputs, user_inputs], dim=1).float()
        dout = torch.nn.functional.relu(self.Layer1(vector))
        dout = torch.nn.functional.relu(self.Layer3(dout))
        dout = torch.nn.functional.relu(self.Layer4(dout))
        sigmoid = torch.nn.Sigmoid()
        prediction = sigmoid(self.Layer5(dout))
        prediction = prediction.to(torch.float32)
        y = prediction.shape[0]
        prediction = prediction.reshape(y)
        return prediction

        # 这就是模型的输出，output，用来预测答案。

    def get_reg(self):
        return torch.sqrt(abs(self.Layer4.weight.data)**2).mean() + torch.sqrt(abs(self.Layer3.weight.data)**2).mean()