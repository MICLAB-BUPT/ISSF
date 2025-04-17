import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
import torch.nn.init as torch_init
from torch.autograd import Variable
from arch.saliency import Saliency_feat_infer
from core.prediction import PredictionModel


def weights_init_random(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)

def calculate_l1_norm(f):
    f_norm = torch.norm(f, p=2, dim=-1, keepdim=True)
    f = f / (f_norm + 1e-9)
    return f



class ISSF(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.w = args.w
        self.in_f = args.input_feat
        self.out_f = args.output_feat
        self.reduce = args.reduce
        self.randw = args.we
        self.lamb = args.lamb
        self.class_n = args.class_num
        self.scale_factor = args.scale_factor
        self.dropout = args.dropout

        self.saliency = nn.Parameter(torch.randn(self.randw, self.out_f))
        self.saliency_back = nn.Parameter(torch.randn(1, self.out_f))
        torch_init.xavier_uniform_(self.saliency)
        self.ac_center = nn.Parameter(torch.randn(self.class_n + 1, self.out_f))
        torch_init.xavier_uniform_(self.ac_center)
        self.fg_center = nn.Parameter(-1.0 * self.ac_center[-1, ...][None, ...])

        self.feature_embedding = nn.Sequential(
            nn.Linear(self.in_f, self.out_f),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
        )

        self.sigmoid = nn.Sigmoid()
        self.apply(weights_init_random)
        self.saliency_feat_infer = Saliency_feat_infer(p=0.5)
        self.weight_linear = nn.Linear(in_features=self.out_f, out_features=self.randw, bias=False)
        self.channel_fore_factors = nn.Parameter(torch.zeros(1), requires_grad=True).cuda()
        self.channel_weights = nn.Sequential(
            nn.Linear(in_features=self.out_f, out_features=self.out_f // self.reduce, bias=False),
            nn.ReLU(),
            nn.Linear(in_features=self.out_f // self.reduce, out_features=self.out_f, bias=False)
        )
        self.channel_back_factors = nn.Parameter(torch.zeros(1), requires_grad=True).cuda()
        self.channel_back_weights = nn.Sequential(
            nn.Linear(in_features=self.out_f, out_features=self.out_f // self.reduce, bias=False),
            nn.ReLU(),
            nn.Linear(in_features=self.out_f // self.reduce, out_features=self.out_f, bias=False)
        )

        self.PredictionModel = PredictionModel(self.ac_center, self.fg_center, self.scale_factor)


    def forward(self, x):
        n, t, _ = x.size()
        x = self.feature_embedding(x)
        saliencyf = self.saliency_feat_infer.generate_saliency_x(x=x, max_class=20)

        saliency_feat_idx = torch.nonzero(saliencyf > 0.)[:, 1].view(-1)
        saliency_feat = x[:, saliency_feat_idx, :]
        saliency = saliency_feat
        saliency_feat = saliency_feat + F.softmax(self.channel_weights(saliency_feat), dim=-1) * saliency_feat
        sim_sa_x = F.softmax(x.bmm(saliency_feat.permute(0, 2, 1)), dim=-1)
        saliency_feat = sim_sa_x.bmm(saliency_feat)

        unsaliency_feat_idx = torch.nonzero(saliencyf < 0.)[:, 1].view(-1)
        unsaliency_feat = x[:, unsaliency_feat_idx, :]
        unsaliency_feat = unsaliency_feat + F.softmax(self.channel_back_weights(unsaliency_feat), dim=-1) * unsaliency_feat
        sim_usa_x = F.softmax(x.bmm(unsaliency_feat.permute(0, 2, 1)), dim=-1)
        unsaliency_feat = sim_usa_x.bmm(unsaliency_feat)

        b_vid_ca_pred, b_vid_cw_pred, b_att, b_frm_pred = self.PredictionModel.PredictionModule(x)

        saliency_feat = (1. - self.lamb) * saliency_feat + self.lamb * unsaliency_feat
        s_vid_ca_pred, s_vid_cw_pred, s_att, s_frm_pred = self.PredictionModel.PredictionModule(saliency_feat)

        norms_saliency = calculate_l1_norm(saliency)
        norms_ac = calculate_l1_norm(self.ac_center)
        saliency_scr = torch.einsum('nkd,cd->nkc', [norms_saliency, norms_ac]) * self.scale_factor
        saliency_pred = F.softmax(saliency_scr, -1)

        return [b_vid_ca_pred, b_vid_cw_pred, b_att, b_frm_pred],\
               [s_vid_ca_pred, s_vid_cw_pred, s_att, s_frm_pred],\
               [saliency_feat, saliency, saliency_pred]
