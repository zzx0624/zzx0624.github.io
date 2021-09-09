import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional
from torch.nn.init import xavier_normal_



class Model(nn.Module):
    def __init__(self, d, tv, tq, to, layers=[64, 32, 16, 8]):
        super(Model, self).__init__()
        # Modules
        self.loss = torch.nn.BCELoss()

        self.MLP_Embedding_User = torch.nn.Embedding(200, layers[0])
        self.Layer1 = torch.nn.Linear(320, layers[1])
        self.Layer3 = torch.nn.Linear(layers[1], layers[2])
        self.Layer4 = torch.nn.Linear(layers[2], layers[3])
        self.Layer5 = torch.nn.Linear(layers[3], 1)
        self.input_dropout = torch.nn.Dropout(0.3)
        self.hidden_dropout1 = torch.nn.Dropout(0.3)
        self.hidden_dropout2 = torch.nn.Dropout(0.3)
        self.bn0 = torch.nn.BatchNorm1d(tq)
        self.bn1 = torch.nn.BatchNorm1d(tv)

    def init(self):
        xavier_normal_(self.MLP_Embedding_User.weight.data)
        # xavier_normal_(self.MLP_Embedding_Q.weight.data)
        # xavier_normal_(self.MLP_Embedding_V.weight.data)
        # xavier_normal_(self.MLP_Embedding_Item.weight.data)

    def forward(self, q_input, v_input, user_input):
        user = self.MLP_Embedding_User(user_input)
        # item = self.MLP_Embedding_Item(item_input)
        # print(user)
        # print(item)
        user_inputs = user.view(user_input.size(0), -1)

        item_inputs = torch.cat([q_input, v_input], dim=1)
        # vector = user_inputs+item_inputs
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

    def get_reg(self):
        return torch.sqrt(abs(self.Wo)**2).mean() + \
               torch.sqrt(abs(self.Wv)**2).mean() + torch.abs(self.Wq).mean()
               # torch.sqrt(abs(self.MLP_Embedding_User.weight.data)**2).mean() + torch.sqrt(abs(self.Q.weight.data)**2).mean() + \
               # torch.sqrt(abs(self.V.weight.data)**2).mean() + torch.sqrt(abs(self.Layer1.weight.data)**2).mean() + \
               # torch.sqrt(abs(self.Layer2.weight.data)**2).mean() + torch.sqrt(abs(self.Layer3.weight.data)**2).mean() + \
               # torch.sqrt(abs(self.Layer4.weight.data)**2).mean()


