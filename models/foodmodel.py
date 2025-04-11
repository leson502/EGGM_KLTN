import torch
import torch.nn as nn
from modules.transformer import TransformerEncoder
import torch.nn.functional as F
from .moe import MoE, Gate
from src.eval_metrics import eval_food

class FoodModel(nn.Module):
    def __init__(self, output_dim=101, num_heads=8, layers=4,
                 relu_dropout=0.1, embed_dropout=0.3, res_dropout=0.1, out_dropout=0.1,
                 attn_dropout=0.25):
        super(FoodModel, self).__init__()
        self.num_mod = 2
        self.proj_dim = 768
        self.num_heads = num_heads
        self.layers = layers
        self.attn_dropout = attn_dropout
        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.out_dropout = out_dropout
        self.embed_dropout = embed_dropout
        self.projv = nn.Linear(768, self.proj_dim)
        self.projt = nn.Linear(768, self.proj_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=self.proj_dim, nhead=self.num_heads, dim_feedforward=self.proj_dim * 4,
            dropout=self.attn_dropout
        )

        self.vision_encoder = nn.TransformerEncoder(
            layer, num_layers=self.layers
        )

        self.text_encoder = nn.TransformerEncoder(
            layer, num_layers=self.layers
        )

        # Output layers
        self.classifier = Classifier(self.proj_dim, output_dim, num_expert=16, num_mod=self.num_mod, k=12)

    def forward(self, v, t):
        v_rep = self.projv(v)
        t_rep = self.projt(t)
        v_rep = self.vision_encoder(v_rep)
        t_rep = self.text_encoder(t_rep)

        out_v, gates_v, load_v = self.classifier(v_rep, 0)
        out_t, gates_t, load_t = self.classifier(t_rep, 1)
        
        out = out_t + out_v

        gates = [gates_v, gates_t]
        load = [load_v, load_t]

        hs = [v_rep.clone().detach(), t_rep.clone().detach()]
        
        return out, hs, [gates, load]



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
            Classifier(768, 101)
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
            acc = eval_food(y, cls_res[i])
            acc_list.append(acc)

        return acc_list

    def forward(self, x):
        self.cls_res = list()
        gatting = 0
        for i in range(len(x)):
            out, _, _ = self.classifers[i](x[i], i)
            self.cls_res.append(out)
        return self.cls_res
    

if __name__ == "__main__":
    model = FoodModel()
    print(model)
    x = torch.randn(2, 768)
    y = torch.randn(2, 1024)
    out, hs, gate_load = model(x, y)
    print(out.shape, hs[0].shape, hs[1].shape)