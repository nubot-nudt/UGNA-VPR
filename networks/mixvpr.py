import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import cirtorch.functional as LF
import math
import torch.nn.functional as F
from einops.layers.torch import Rearrange, Reduce
import torchvision.models as models
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from torch.optim import lr_scheduler, optimizer
import utils
from torch import Tensor
# from dataloaders.GSVCitiesDataloader import GSVCitiesDataModule
from networks.models import helper

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
class VPRModel(pl.LightningModule):
    """This is the main model for Visual Place Recognition
    we use Pytorch Lightning for modularity purposes.

    Args:
        pl (_type_): _description_
    """

    def __init__(self,
                 # ---- Backbone
                 backbone_arch='resnet50',
                 pretrained=True,
                 layers_to_freeze=1,
                 layers_to_crop=[],

                 # ---- Aggregator
                 agg_arch='ConvAP',  # CosPlace, NetVLAD, GeM
                 agg_config={},

                 # ---- Train hyperparameters
                 lr=0.03,
                 optimizer='sgd',
                 weight_decay=1e-3,
                 momentum=0.9,
                 warmpup_steps=500,
                 milestones=[5, 10, 15],
                 lr_mult=0.3,

                 # ----- Loss
                 loss_name='MultiSimilarityLoss',
                 miner_name='MultiSimilarityMiner',
                 miner_margin=0.1,
                 faiss_gpu=False
                 ):
        super().__init__()
        self.encoder_arch = backbone_arch
        self.pretrained = pretrained
        self.layers_to_freeze = layers_to_freeze
        self.layers_to_crop = layers_to_crop

        self.agg_arch = agg_arch
        self.agg_config = agg_config

        self.lr = lr
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.warmpup_steps = warmpup_steps
        self.milestones = milestones
        self.lr_mult = lr_mult

        self.loss_name = loss_name
        self.miner_name = miner_name
        self.miner_margin = miner_margin

        self.save_hyperparameters()  # write hyperparams into a file

        #self.loss_fn = utils.get_loss(loss_name)
        #self.miner = utils.get_miner(miner_name, miner_margin)
        self.batch_acc = []  # we will keep track of the % of trivial pairs/triplets at the loss level

        self.faiss_gpu = faiss_gpu

        # ----------------------------------
        # get the backbone and the aggregator
        self.backbone = helper.get_backbone(backbone_arch, pretrained, layers_to_freeze, layers_to_crop)
        self.aggregator = helper.get_aggregator(agg_arch, agg_config)

    # the forward pass of the lightning model
    def forward(self, x):
        x = self.backbone(x)
        feature = x
        x = self.aggregator(x)
        return x, feature


class Shen(nn.Module): #整合Vit和resnet
    def __init__(self, opt=None):
        super().__init__()
        self.backbone = VPRModel(
            # ---- Encoder
            backbone_arch='resnet50',
            pretrained=True,
            layers_to_freeze=2,
            layers_to_crop=[4],  # 4 crops the last resnet layer, 3 crops the 3rd, ...etc

            # ---- Aggregator
            # agg_arch='CosPlace',
            # agg_config={'in_dim': 2048,
            #             'out_dim': 2048},
            # agg_arch='GeM',
            # agg_config={'p': 3},

            # agg_arch='ConvAP',
            # agg_config={'in_channels': 2048,
            #             'out_channels': 2048},

            agg_arch='MixVPR',
            agg_config={'in_channels': 1024,
                        'in_h': 14,
                        'in_w': 14,
                        'out_channels': 128,
                        'mix_depth': 4,
                        'mlp_ratio': 1,
                        'out_rows': 4},  # the output dim will be (out_rows * out_channels)

            # ---- Train hyperparameters
            lr=0.05,  # 0.0002 for adam, 0.05 or sgd (needs to change according to batch size)
            optimizer='sgd',  # sgd, adamw
            weight_decay=0.001,  # 0.001 for sgd and 0 for adam,
            momentum=0.9,
            warmpup_steps=650,
            milestones=[5, 10, 15, 25, 45],
            lr_mult=0.3,

            # ----- Loss functions
            # example: ContrastiveLoss, TripletMarginLoss, MultiSimilarityLoss,
            # FastAPLoss, CircleLoss, SupConLoss,
            loss_name='MultiSimilarityLoss',
            miner_name='MultiSimilarityMiner',  # example: TripletMarginMiner, MultiSimilarityMiner, PairMarginMiner
            miner_margin=0.1,
            faiss_gpu=False
        )
    def forward(self, inputs):
        out, feature=self.backbone(inputs) #(B,S,C)

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
        #mu = self.mean_head(backbone_output).view(B, -1)                                           # ([B, 2048]) <= ([B, 2048, 1, 1])

        return backbone_output, shen


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