import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import pfrl
from pfrl import experiments, utils
from torch.distributions.categorical import Categorical
# from multi_categorical import Categorical_multi
from pfrl.policies import SoftmaxCategoricalHead

class Network_policy(nn.Module):
    def __init__(self):
        super(Network_policy, self).__init__()
        # super().__init__()

        self.mlp_emd_d1 = nn.Linear(3, 16)
        self.mlp_emd_d2 = nn.Linear(16, 16)
        self.mlp_emd_o1 = nn.Linear(6, 16)
        self.mlp_emd_o2 = nn.Linear(16, 16)

        self.mlp_w_d1 = nn.Linear(16, 16)
        self.mlp_w_d2 = nn.Linear(16, 1)
        self.mlp_w_o1 = nn.Linear(16, 16)
        self.mlp_w_o2 = nn.Linear(16, 1)

        
        self.mlp_assign = nn.Linear(64, 8) 
        self.final_layer = nn.Linear(8,1)

        nn.init.orthogonal_(self.mlp_emd_d1.weight, 1)
        nn.init.orthogonal_(self.mlp_emd_d2.weight, 1)
        nn.init.orthogonal_(self.mlp_emd_o1.weight, 1)
        nn.init.orthogonal_(self.mlp_emd_o2.weight, 1)
        nn.init.orthogonal_(self.mlp_w_d1.weight, 1)
        nn.init.orthogonal_(self.mlp_w_d2.weight, 1)
        nn.init.orthogonal_(self.mlp_w_o1.weight, 1)
        nn.init.orthogonal_(self.mlp_w_o2.weight, 1)
        nn.init.orthogonal_(self.mlp_assign.weight, 1)
        nn.init.orthogonal_(self.final_layer.weight, 1e-2)

        nn.init.zeros_(self.mlp_emd_d1.bias)
        nn.init.zeros_(self.mlp_emd_d2.bias)
        nn.init.zeros_(self.mlp_emd_o1.bias)
        nn.init.zeros_(self.mlp_emd_o2.bias)
        nn.init.zeros_(self.mlp_w_d1.bias)
        nn.init.zeros_(self.mlp_w_d2.bias)
        nn.init.zeros_(self.mlp_w_o1.bias)
        nn.init.zeros_(self.mlp_w_o2.bias)
        nn.init.zeros_(self.mlp_assign.bias)
        nn.init.zeros_(self.final_layer.bias)



    def forward(self, obs):

        tasks_no = 10
        agent_no = 100
        

        inshape = np.shape(obs)
        obs = obs.reshape((-1,3*(agent_no+1)+(6*tasks_no)))
        tasks_reshaped = obs[:,0:6*tasks_no].reshape((-1,tasks_no,6)) #changed

        pos_reshaped = obs[:,6*tasks_no:(6*tasks_no+3*agent_no)].reshape((-1,agent_no,3))
        robot_selected = obs[:,(6*tasks_no+3*agent_no):(6*tasks_no+3*agent_no)+3].reshape((-1,3))

        v_d0 = nn.ReLU()(self.mlp_emd_d1(pos_reshaped))
        v_d = self.mlp_emd_d2(v_d0)
        alpha_d0 = torch.tanh(self.mlp_w_d1(v_d))
        alpha_d = nn.Sigmoid()(self.mlp_w_d2(alpha_d0))
        driver = torch.sum(alpha_d*v_d, dim=1)

        
        robot_selected0 = nn.ReLU()(self.mlp_emd_d1(robot_selected))
        robot_selected1 = self.mlp_emd_d2(robot_selected0)
        
        v_o0 = nn.ReLU()(self.mlp_emd_o1(tasks_reshaped))
        v_o = self.mlp_emd_o2(v_o0)
        alpha_o0 = torch.tanh(self.mlp_w_o1(v_o))
        alpha_o = nn.Sigmoid()(self.mlp_w_o2(alpha_o0))
        order = torch.sum(alpha_o*v_o, dim=1) 
        
        vector = torch.cat([driver, order, robot_selected1], dim=1).reshape((-1,1,48))
        vector = vector.repeat(1, tasks_no, 1) #.reshape((-1, 2, 48))  # changed 20 -> 5
        vector = torch.cat([v_o, vector], dim=2)
        
        output = nn.ReLU()(self.mlp_assign(vector))
        output = torch.squeeze(self.final_layer(output))   #.reshape((-1, 2))  # Relu removed
        # output = SoftmaxCategoricalHead()(output)          

        return output