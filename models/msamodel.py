import torch
from torch import nn
import torch.nn.functional as F
from src.eval_metrics import *
from .moe import MoE, Gate
from torch.nn.utils.rnn import unpad_sequence
from modules.transformer import TransformerEncoder


class MSAModel(nn.Module):
    def __init__(self, output_dim, orig_dim, proj_dim=512, num_heads=5, layers=5,
                 relu_dropout=0.1, embed_dropout=0.3, res_dropout=0.1, out_dropout=0.1,
                 attn_dropout=0.25
                 ):
        super(MSAModel, self).__init__()

        self.proj_dim = proj_dim
        self.orig_dim = orig_dim
        self.num_mod = len(orig_dim)
        self.num_heads = num_heads
        self.layers = layers
        self.attn_dropout = attn_dropout
        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.out_dropout = out_dropout
        self.embed_dropout = embed_dropout

        # Projection Layers
        self.proj = nn.ModuleList([
            nn.Conv1d(self.orig_dim[i], self.proj_dim, kernel_size=1, padding=0)
            for i in range(self.num_mod)
        ])

        # Encoders
        self.encoders = nn.ModuleList([
            TransformerEncoder(embed_dim=proj_dim, num_heads=self.num_heads,
                               layers=self.layers, attn_dropout=self.attn_dropout, res_dropout=self.res_dropout,
                               relu_dropout=self.relu_dropout, embed_dropout=self.embed_dropout)
            for _ in range(self.num_mod)
        ])

        # Fusion
        self.classifier = Classifier(self.proj_dim, output_dim, num_expert=16, num_mod=self.num_mod, k=4)


    def forward(self, x, lengths):
        """
        dimension [batch_size, seq_len, n_features]
        """
        hs = list()
        hs_detach = list()
        output = 0
        gates = []
        loads = []
        for i in range(self.num_mod):
            x[i] = x[i].transpose(1, 2)
            x[i] = self.proj[i](x[i])
            x[i] = x[i].permute(2, 0, 1)
            h_tmp = self.encoders[i](x[i]) # [seq_len, batch_size, proj_dim]
            h_tmp = torch.concat(unpad_sequence(h_tmp, lengths.cpu()))
            hs.append(h_tmp)
            hs_detach.append(h_tmp.clone().detach())

            out_tmp, gate, load = self.classifier(h_tmp, i)
            gates.append(gate)
            loads.append(load)
            output += out_tmp
        
        return output, hs_detach, [gates, loads]



class Classifier(nn.Module):
    def __init__(self, in_dim, out_dim, num_expert=16, num_mod=2, k=12):
        super(Classifier, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_expert = num_expert
        self.num_mod = num_mod
        self.k = k

        self.moe = MoE(in_dim, in_dim, num_expert, in_dim // 4, True)
        self.gatting_network = nn.ModuleList(Gate(in_dim, num_expert, k=k) for _ in range(num_mod))
        self.out_layer = nn.Linear(in_dim, out_dim)
    
    def forward(self, x, modality):
        gates, load = self.gatting_network[modality](x, modality)
        
        out = self.moe(x, gates)
        x = F.relu(out) + x
        x = self.out_layer(x)
        return x, gates, load


class ClassifierGuided(nn.Module):
    def __init__(self, num_mod, proj_dim=1024):
        super(ClassifierGuided, self).__init__()
        # Classifiers
        self.num_mod = num_mod
        self.classifers = nn.ModuleList([
            Classifier(512, 2, num_expert=16, num_mod=num_mod, k=4)
            for _ in range(self.num_mod)
        ])
    
    def init_classifier(self, cls):
        for i in range(self.num_mod):
            with torch.no_grad():
                
                for param_A, param_B in zip(self.classifers[i].parameters(), cls.parameters()):
                    param_A.data.copy_(param_B.data)


    def cal_coeff(self, y, cls_res):
        acc_list = list()
        for i in range(self.num_mod):
            acc = train_eval_food(y, cls_res[i])
            acc_list.append(acc)

        return acc_list

    def forward(self, x):
        self.cls_res = list()
        gatting = 0
        for i in range(len(x)):
            out, _, _ = self.classifers[i](x[i], i)
            self.cls_res.append(out)
        return self.cls_res
    