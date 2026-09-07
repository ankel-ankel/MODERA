import torch
import torch.nn as nn
import torch.nn.functional as F


# các loss thử nghiệm, cấu hình chính chỉ dùng CE
class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
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
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, anomaly_id: int = 0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.anomaly_id = anomaly_id

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, labels, reduction="none")
        pt = torch.exp(-ce)
        alpha_t = torch.where(labels == self.anomaly_id, self.alpha, 1 - self.alpha)
        return (alpha_t * (1 - pt) ** self.gamma * ce).mean()


class LogitAdjustedCE(nn.Module):
    def __init__(self, base_probs: list[float], tau: float = 1.0, label_smoothing: float = 0.0):
        super().__init__()
        log_prior = torch.log(torch.as_tensor(base_probs, dtype=torch.float32) + 1e-12)
        self.register_buffer("log_prior", log_prior)
        self.tau = tau
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        adjusted = logits + self.tau * self.log_prior.to(logits.device).to(logits.dtype)
        return F.cross_entropy(adjusted, labels, label_smoothing=self.label_smoothing)


class CombinedLoss(nn.Module):
    def __init__(self, alpha_supcon: float = 0.0, supcon_temperature: float = 0.07,
                 label_smoothing: float = 0.0, loss_type: str = "ce",
                 focal_gamma: float = 2.0, focal_alpha: float = 0.25,
                 base_probs: list[float] | None = None, la_tau: float = 1.0,
                 anomaly_id: int = 0):
        super().__init__()
        self.alpha_supcon = alpha_supcon
        self.loss_type = loss_type
        if loss_type == "focal":
            self.cls_loss = FocalLoss(gamma=focal_gamma, alpha=focal_alpha, anomaly_id=anomaly_id)
        elif loss_type == "logit_adj":
            assert base_probs is not None, "logit_adj requires base_probs"
            self.cls_loss = LogitAdjustedCE(base_probs=base_probs, tau=la_tau, label_smoothing=label_smoothing)
        else:
            self.cls_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.supcon = SupConLoss(temperature=supcon_temperature)

    def forward(self, logits: torch.Tensor, embeddings: torch.Tensor, labels: torch.Tensor,
                projection: torch.Tensor | None = None):
        cls = self.cls_loss(logits, labels)
        if self.alpha_supcon > 0.0:
            contrast_feat = projection if projection is not None else F.normalize(embeddings, dim=-1)
            sc = self.supcon(contrast_feat, labels)
            total = cls + self.alpha_supcon * sc
            return total, {"cls": cls.item(), "supcon": sc.item(), "total": total.item()}
        return cls, {"cls": cls.item(), "supcon": 0.0, "total": cls.item()}
