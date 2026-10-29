import torch
from torch import nn

def features(w, variant):
    if w.shape[-2:] != (8,6): raise ValueError('Expected 8x6 cycle')
    if variant == 'f1': return w[..., -1:, :].expand_as(w).reshape(*w.shape[:-2],48)
    if variant == 'f2':
        peak=w.gather(-2,w.abs().argmax(-2,keepdim=True)).squeeze(-2)
        x=torch.cat((w[...,-1,:],w.mean(-2),w.std(-2,unbiased=False),peak),-1)
        return torch.cat((x,torch.zeros_like(x)),-1)
    if variant == 'f3': return w.flatten(-2)
    raise ValueError(variant)


class CycleEncoder(nn.Module):
    def __init__(self, cfg, backbone):
        super().__init__(); self.variant=cfg.force_cycle_variant
        self.force=nn.Sequential(nn.Linear(48,64),nn.Mish(),nn.Linear(64,64),nn.Mish())
        self.backbone=backbone
    def forward(self,x):
        # WorldModel appends task embedding after state observations.
        f=features(x[...,14:62].reshape(*x.shape[:-1],8,6),self.variant)
        return self.backbone(torch.cat((x[...,:14],self.force(f),x[...,62:]),-1))
