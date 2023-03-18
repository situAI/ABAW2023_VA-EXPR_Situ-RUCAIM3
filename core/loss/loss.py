import torch
import torch.nn.functional as F
import torch.nn as nn

from utils.registery import LOSS_REGISTRY

@LOSS_REGISTRY.register()
class CCCLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        y = y.contiguous().view(-1)
        x = x.contiguous().view(-1)
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)
        rho = torch.sum(vx * vy) / (
            torch.sqrt(torch.sum(torch.pow(vx, 2))) * torch.sqrt(torch.sum(torch.pow(vy, 2))) + self.eps)
        x_m = torch.mean(x)
        y_m = torch.mean(y)
        x_s = torch.std(x)
        y_s = torch.std(y)
        ccc = 2 * rho * x_s * y_s / (torch.pow(x_s, 2) + torch.pow(y_s, 2) + torch.pow(x_m - y_m, 2))
        return 1 - ccc

@LOSS_REGISTRY.register()
class VALoss(nn.Module):
    def __init__(self, alpha=1, beta=1, eps=1e-8):
        super().__init__()
        self.ccc = CCCLoss(eps=eps)
        self.alpha = alpha
        self.beta = beta

    def forward(self, x, y):
        loss = self.alpha * self.ccc(x[:, 0], y[:, 0]) + self.beta * self.ccc(x[:, 1], y[:, 1])

        return loss

@LOSS_REGISTRY.register()
class ExprLoss(nn.Module):
    def __init__(self, weight):
        super().__init__()
        w = torch.FloatTensor(weight)
        self.ce = nn.CrossEntropyLoss(weight=w)

    def forward(self, x, y):
        return self.ce(x, y)


@LOSS_REGISTRY.register()
class RDropLoss(nn.Module):
    def __init__(self, weight, alpha=5):
        super().__init__()
        w = torch.FloatTensor(weight)
        self.ce = nn.CrossEntropyLoss(weight=w, reduction='none')
        self.kld = nn.KLDivLoss(reduction='none')
        self.alpha = alpha

    def forward(self, logits1, logits2, gt):
        ce_loss = (self.ce(logits1, gt) + self.ce(logits2, gt)) / 2
        kl_loss1 = self.kld(F.log_softmax(logits1, dim=-1), F.softmax(logits2, dim=-1)).sum(-1)
        kl_loss2 = self.kld(F.log_softmax(logits2, dim=-1), F.softmax(logits1, dim=-1)).sum(-1)
        kl_loss = (kl_loss1 + kl_loss2) / 2
        loss = ce_loss + self.alpha * kl_loss

        loss = loss.mean(-1)

        return loss
