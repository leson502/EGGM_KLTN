import torch
import torch.nn as nn
import torch.nn.functional as F
from .moe import MoE, Gate
from copy import deepcopy

from src.eval_metrics import eval_crema
from transformers import ViTModel, ASTForAudioClassification

class CREMADModel(nn.Module):
    def __init__(self):
        super(CREMADModel, self).__init__()

        self.n_classes = 6
        ast_model = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.encoder_0 = ast_model.audio_spectrogram_transformer
        self.encoder_1 = ViTModel.from_pretrained("google/vit-base-patch16-224")
        
        self.classifier = Classifier(768, self.n_classes)

    def forward(self, audio, visual):
        a = self.encoder_0(audio).pooler_output
        v = self.encoder_1(visual).pooler_output
        
        out_a, gates_a, load_a = self.classifier(a, 0)
        out_v, gates_v, load_v = self.classifier(v, 1)

        gates = [gates_a, gates_v]
        load = [load_a, load_v]
     
        hs = [a.clone().detach(), v.clone().detach()]
        out = out_a + out_v

        return out, hs, [gates, load]

def cv_squared(x):
    """The squared coefficient of variation of a sample.
    Useful as a loss to encourage a positive distribution to be more uniform.
    Epsilons added for numerical stability.
    Returns 0 for an empty Tensor.
    Args:
    x: a `Tensor`.
    Returns:
    a `Scalar`.
    """
    eps = 1e-10
    # if only num_experts = 1

    if x.shape[0] == 1:
        return torch.tensor([0], device=x.device, dtype=x.dtype)
    return x.float().var() / (x.float().mean()**2 + eps)

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
            Classifier(768, 6)
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
            acc = eval_crema(y, cls_res[i])
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
    model = CREMADModel()
    print(model)
    audio = torch.randn(1, 1, 256, 128)
    visual = torch.randn(1, 3, 224, 224)
    out, hs = model(audio, visual)
    print(out.size())
    print(len(hs))
    print(hs[0].size())
    print(hs[1].size())