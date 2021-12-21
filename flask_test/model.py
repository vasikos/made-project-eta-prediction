import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
from dgl.nn import GraphConv

class Model(nn.Module):
  def __init__(self, idim, hdim1, hdim2):
    super().__init__()
    #self.conv1 = GraphConv(idim, hdim1, activation=F.relu)
    self.conv1 = GraphConv(idim, hdim1)
    self.conv2 = GraphConv(hdim1, hdim1, activation=F.relu)
    self.fc1 = nn.Linear(hdim1, hdim2)
    self.fc2 = nn.Linear(hdim2, hdim2)
    self.fc3 = nn.Linear(hdim2, 1)
    self.bn2 = nn.BatchNorm1d(hdim2)
    self.bn1 = nn.BatchNorm1d(hdim1)
    self.relu = nn.ReLU()

  def forward(self, g, node_feat, batch):
    vec = self.conv1(g, g.ndata[node_feat])
    vec = self.bn1(vec)
    vec = F.leaky_relu(vec)
    #vec = F.dropout(vec, 0.1)
    vec = self.conv2(g, vec)
    vec1 = vec[batch[0]]
    vec2 = vec[batch[1]]
    emb = vec1 - vec2
    emb = self.fc1(emb)
    emb = self.bn2(emb)
    emb = F.leaky_relu(emb)
    #emb = F.dropout(emb, 0.1)
    emb = F.leaky_relu(self.fc2(emb))
    emb = self.fc3(emb)
    return emb

def load_model():
    tmp_model = Model(2,16,16)
    '''
    try:
        tmp_model = torch.load('model.pkl')
        print('model loaded')
    except:
        print("can't find the model")
    '''    
    #tmp_model = torch.load('model.pkl')
    tmp_model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))

    return tmp_model

