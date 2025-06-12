import torch
from torch.nn import functional as F
import numpy as np
import torch.nn as nn
import torch.nn.init as torch_init


class Saliency_feat_infer(object):
    def __init__(self, p=0.1):
        self.p = p
        self.conv = nn.Conv1d(in_channels=2048, out_channels=2048, kernel_size=3, padding=1).cuda()

    def generate_saliency_x(self, x, max_class):
        batch, num_frames, dim = x.shape
        saliency_infer = torch.zeros(size=(batch, num_frames)) - 100.
        previous_x = torch.zeros_like(x)
        previous_x[:, 1:, :] = x[:, :-1, :]
        saliency_x = x - previous_x
        saliency_x = torch.abs(saliency_x)
        saliency_x = saliency_x.sum(dim=-1)
        _, num_saliency = saliency_x.shape
        median = int(num_saliency * self.p) - 1
        saliency_candidates = torch.topk(saliency_x, k=median, largest=True)[1]

        for b in range(saliency_candidates.size(0)):
            selected_clicks = saliency_candidates[b]
            saliency_infer[b, selected_clicks-1] = float(max_class)
            saliency_infer[b, selected_clicks] = float(max_class)
            saliency_infer = saliency_infer.long()

        return saliency_infer.cuda()



class Saliency_Memory(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.sa_nu = args.sa_nu
        self.class_n = args.class_num
        self.out_f = args.output_feat
        self.moment_up = 85
        self.Epoch = args.epoch
        self.m = 0.0
        self.register_buffer("cls_sa_queue", torch.zeros(self.class_n, self.sa_nu, self.out_f))
        self.register_buffer("cls_sa_sc_queue", torch.zeros(self.class_n, self.sa_nu))


    @torch.no_grad()
    def _update_queue(self, inp_sa, inp_sa_sc, cls_idx, epoch):
        for idx in cls_idx:
            if epoch <= self.moment_up:
                self._sort_permutation(inp_sa, inp_sa_sc, idx)
            else:
                t_div_T = float(epoch) / float(self.Epoch)
                assert t_div_T >= 0
                self.m = 0.5 * np.log(np.exp(t_div_T) + 1)
                self._moment_sort_permutation(inp_sa, inp_sa_sc, idx)

    @torch.no_grad()
    def _sort_permutation(self, inp_sa, inp_sa_sc, idx):
        concat_sa_sc = torch.cat([self.cls_sa_sc_queue[idx, ...], inp_sa_sc[..., idx]], 0)
        concat_sa = torch.cat([self.cls_sa_queue[idx, ...], inp_sa], 0)
        sorted_sa_sc, indices = torch.sort(concat_sa_sc, descending=True)
        sorted_sa = torch.index_select(concat_sa, 0, indices[:self.sa_nu])
        self.cls_sa_queue[idx, ...] = sorted_sa
        self.cls_sa_sc_queue[idx, ...] = sorted_sa_sc[:self.sa_nu]

    @torch.no_grad()
    def _moment_sort_permutation(self, inp_sa, inp_sa_sc, idx):
        concat_sa_sc = torch.cat([self.cls_sa_sc_queue[idx, ...], inp_sa_sc[..., idx]], 0)
        concat_sa = torch.cat([self.cls_sa_queue[idx, ...], inp_sa], 0)
        sorted_sa_sc, indices = torch.sort(concat_sa_sc, descending=True)
        sorted_sa = torch.index_select(concat_sa, 0, indices[:self.sa_nu])
        self.cls_sa_queue[idx, ...] = self.m * self.cls_sa_queue[idx, ...] + (1 - self.m) * sorted_sa
        self.cls_sa_sc_queue[idx, ...] = sorted_sa_sc[:self.sa_nu]

    @torch.no_grad()
    def _init_queue(self, sa_queue, sa_sc_queue, lbl_queue, epoch):
        for sa, sa_sc, lbl in zip(sa_queue, sa_sc_queue, lbl_queue):
            idxs = np.where(lbl==1)[0].tolist()
            self._update_queue(sa, sa_sc, idxs, epoch)

    @torch.no_grad()
    def _return_queue(self, cls_idx):
        sas = []
        for idx in cls_idx:
            sas.append(self.cls_sa_queue[idx][None, ...])
        sas = torch.cat(sas, 1)
        return sas


