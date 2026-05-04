import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        features = F.normalize(features, dim=1)
        labels = labels.view(-1, 1)
        same_class = torch.eq(labels, labels.T).float()

        logits = features @ features.T / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        not_self = 1.0 - torch.eye(features.size(0), device=device)
        same_class = same_class * not_self

        exp_logits = torch.exp(logits) * not_self
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        pos_count = same_class.sum(1)
        valid = pos_count > 0
        if not valid.any():
            return torch.zeros((), device=device, requires_grad=True)
        return -(same_class * log_prob).sum(1)[valid].div(pos_count[valid]).mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, labels):
        log_p = F.log_softmax(logits, dim=-1)
        if self.label_smoothing > 0:
            n = logits.size(-1)
            target = torch.full_like(log_p, self.label_smoothing / (n - 1))
            target.scatter_(-1, labels.unsqueeze(-1), 1.0 - self.label_smoothing)
            ce = -(target * log_p).sum(dim=-1)
        else:
            ce = F.nll_loss(log_p, labels, reduction="none")
        pt = log_p.gather(-1, labels.unsqueeze(-1)).squeeze(-1).exp()
        loss = (1 - pt).pow(self.gamma) * ce
        if self.alpha is not None:
            loss = loss * self.alpha[labels]
        return loss.mean()


class CombinedLoss(nn.Module):
    def __init__(self, alpha_supcon=0.1, supcon_temperature=0.07,
                 class_weights=None, use_focal=False, focal_gamma=2.0,
                 label_smoothing=0.0):
        super().__init__()
        self.alpha_supcon = alpha_supcon
        if use_focal:
            self.cls_loss = FocalLoss(alpha=class_weights, gamma=focal_gamma,
                                      label_smoothing=label_smoothing)
        else:
            self.cls_loss = nn.CrossEntropyLoss(weight=class_weights,
                                                label_smoothing=label_smoothing)
        self.supcon = SupConLoss(temperature=supcon_temperature)

    def forward(self, logits, embeddings, labels):
        cls = self.cls_loss(logits, labels)
        if self.alpha_supcon > 0.0:
            sc = self.supcon(embeddings, labels)
            total = cls + self.alpha_supcon * sc
            return total, {"cls": cls.item(), "supcon": sc.item(), "total": total.item()}
        return cls, {"cls": cls.item(), "supcon": 0.0, "total": cls.item()}
