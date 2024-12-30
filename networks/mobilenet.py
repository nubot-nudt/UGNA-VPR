#%%
import sys

sys.path.append('..')
from re import L
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import cirtorch.functional as LF
import math
import torch.nn.functional as F
import torch
import timm
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
import torchvision.models as models
from netvlad import NetVLADLoupe
from torch import Tensor
from torchvision.models import mobilenet_v2
class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return F.normalize(input, p=2, dim=self.dim)


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            # 使用一个卷积层而不是一个线性层 -> 性能增加
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            # 将卷积操作后的patch铺平
            Rearrange('b e h w -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.projection(x)
        return x


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return LF.gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=1024, dropout=0.1):
        super().__init__()

        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


class Shen(nn.Module): #整合Vit和resnet
    def __init__(self, opt=None):
        super().__init__()
        heads = 4
        d_model = 512
        dropout = 0.1
        mobilenet_v2 = models.mobilenet_v2(pretrained=True)
        featuresmobile = list(mobilenet_v2.children())[:-1]
        #print(featuresmobile)
        self.backbone = nn.Sequential(*featuresmobile)
        #self.backbone = mobilenet_v2
        self.linear = nn.Sequential(
            nn.Flatten(),  # 展平操作
            nn.Dropout(p=0.2),  # Dropout 层
            nn.Linear(in_features=1280 * 7 * 7, out_features=512)  # 全连接层
        )
        #self.net_vlad = NetVLADLoupe(feature_size=49, max_samples=1280, cluster_size=64,
                                     #output_dim=512, gating=True, add_batch_norm=False,
                                     #is_training=True)
    def forward(self, inputs):
        #ViT branch
        out=self.backbone(inputs) #(B,S,C)
        #print(out.shape)
        feature=out
        out=self.linear(out)
        #print(out.shape)
        #out= self.net_vlad(out)

        #feature_V_enhanced = self.net_vlad_V(feature_V)

        return out,feature


class Backbone(nn.Module):
    def __init__(self, opt=None):
        super().__init__()

        self.sigma_dim = 2048
        self.mu_dim = 2048

        self.backbone = Shen()


class Stu_Backbone(nn.Module):
    def __init__(self):
        super(Stu_Backbone, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True)


    def forward(self, inputs):
        #Res branch(1*1024)
        outRR =  self.resnet50(inputs)


        return outRR


class TeacherNet(Backbone):
    def __init__(self, opt=None):
        super().__init__()
        self.id = 'teacher'
        self.mean_head = nn.Sequential(L2Norm(dim=1))

    def forward(self, inputs):
        B, C, H, W = inputs.shape                # (B, 1, 3, 224, 224)
                                                 # inputs = inputs.view(B * L, C, H, W)     # ([B, 3, 224, 224])

        backbone_output,shen = self.backbone(inputs)      # ([B, 2048, 1, 1])
        #print(backbone_output.shape)
        mu = self.mean_head(backbone_output).view(B, -1)                                           # ([B, 2048]) <= ([B, 2048, 1, 1])

        return mu, shen


class StudentNet(TeacherNet):
    def __init__(self, opt=None):
        super().__init__()
        self.id = 'student'
        self.var_head = nn.Sequential(nn.Linear(2048, self.sigma_dim), nn.Sigmoid())
        self.backboneS = Stu_Backbone()
    def forward(self, inputs):
        B, C, H, W = inputs.shape                # (B, 1, 3, 224, 224)
        inputs = inputs.view(B, C, H, W)         # ([B, 3, 224, 224])
        backbone_output = self.backboneS(inputs)

        mu = self.mean_head(backbone_output).view(B, -1)                                           # ([B, 2048]) <= ([B, 2048, 1, 1])
        log_sigma_sq = self.var_head(backbone_output).view(B, -1)                                  # ([B, 2048]) <= ([B, 2048, 1, 1])

        return mu, log_sigma_sq


def deliver_model(opt, id):
    if id == 'tea':
        return TeacherNet(opt)
    elif id == 'stu':
        return StudentNet(opt)


if __name__ == '__main__':
    tea = TeacherNet()
    #stu = StudentNet()
    inputs = torch.rand((1, 3, 224, 224))
    #pretrained_weights_path = '../logs/ckpt_best.pth.tar'
    #pretrained_state_dict = torch.load(pretrained_weights_path)
    #tea.load_state_dict(pretrained_state_dict["state_dict"])
    outputs_tea,shen = tea(inputs)
    print(outputs_tea.shape)
    print(shen.shape)
    #x = model(shen)
    #print(x.shape)
    #print(shen.shape)
    #outputs_stu = stu(inputs)
   # print(outputs_stu.shape)
   # print(tea.state_dict())
    #print(outputs_tea[0].shape, outputs_tea[1].shape)
    #print(outputs_stu[0].shape, outputs_stu[1].shape)
