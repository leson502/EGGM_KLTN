import torch
import torch.nn as nn
from modules.transformer import TransformerEncoder
import torch.nn.functional as F
from .moe import MoE, Gate
from src.eval_metrics import eval_food
from transformers import BertTokenizer, BertModel
from transformers import ViTModel, ViTFeatureExtractor

class FoodModel(nn.Module):
    def __init__(self, output_dim=101):
        super(FoodModel, self).__init__()
        self.num_mod = 2
        self.proj_dim = 768
       
        self.encoder_0 = ViTModel.from_pretrained("google/vit-base-patch16-224")
        self.encoder_1 = BertModel.from_pretrained("google-bert/bert-base-uncased")
        # Output layers
        self.classifier = Classifier(self.proj_dim, output_dim, num_expert=16, num_mod=self.num_mod, k=12)

    def forward(self, v, t):
        v = self.encoder_0(v).pooler_output
        t = self.encoder_1(t).pooler_output

        out_v, gates_v, load_v = self.classifier(v, 0)
        out_t, gates_t, load_t = self.classifier(t, 1)
    
        out = out_t + out_v

        gates = [gates_v, gates_t]
        load = [load_v, load_t]

        hs = [v.clone().detach(), t.clone().detach()]
        
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