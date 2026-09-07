from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel


@dataclass
class LogOutput:
    logits: torch.Tensor
    embeddings: torch.Tensor
    projection: torch.Tensor | None = None


# gộp chuỗi token thành một vector cho mỗi cửa sổ
class AttentionPool(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        scores = self.attn(hidden_states).squeeze(-1)
        scores = scores.masked_fill(~attention_mask.bool(), -1e9)
        weights = F.softmax(scores, dim=1).unsqueeze(-1)
        return (hidden_states * weights).sum(dim=1)


# head thay thế, cấu hình chính không dùng
class MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_labels: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_labels)
        for m in (self.fc1, self.fc2):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.drop(self.act(self.norm(self.fc1(x)))))


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.fc2(self.act(self.fc1(x))), dim=-1)


# encoder + pooling + head phân loại
class ModernBertClassifier(nn.Module):
    def __init__(self, model_path: str, num_labels: int, pooling: str = "mean",
                 attn_implementation: str | None = None, dtype: torch.dtype = torch.bfloat16,
                 use_mlp_head: bool = False, head_hidden: int = 512, head_dropout: float = 0.2,
                 use_proj_head: bool = False, proj_hidden: int = 256, proj_out: int = 128,
                 grad_checkpoint: bool = False):
        super().__init__()
        config = AutoConfig.from_pretrained(model_path)
        if attn_implementation is None:
            try:
                self.encoder = AutoModel.from_pretrained(
                    model_path, attn_implementation="sdpa", dtype=dtype,
                )
                attn_implementation = "sdpa"
            except (ValueError, NotImplementedError):
                self.encoder = AutoModel.from_pretrained(
                    model_path, attn_implementation="eager", dtype=dtype,
                )
                attn_implementation = "eager"
        else:
            self.encoder = AutoModel.from_pretrained(
                model_path, attn_implementation=attn_implementation, dtype=dtype,
            )
        self.attn_implementation = attn_implementation
        self.pooling = pooling
        self.use_proj_head = use_proj_head
        if grad_checkpoint and hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()

        if pooling == "attention":
            self.pool_attn = AttentionPool(config.hidden_size).to(dtype=dtype)
            cls_in = config.hidden_size
        elif pooling == "mean_max":
            cls_in = 2 * config.hidden_size
        else:
            cls_in = config.hidden_size

        if use_mlp_head:
            self.classifier = MLPHead(cls_in, head_hidden, num_labels, head_dropout)
        else:
            self.classifier = nn.Linear(cls_in, num_labels)

        if use_proj_head:
            self.proj_head = ProjectionHead(cls_in, proj_hidden, proj_out)
        else:
            self.proj_head = None

        self.num_labels = num_labels

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> LogOutput:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        h = out.last_hidden_state
        if self.pooling == "attention":
            pooled = self.pool_attn(h, attention_mask)
        elif self.pooling == "mean_max":
            mask = attention_mask.unsqueeze(-1).to(h.dtype)
            mean_p = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
            masked_h = h.masked_fill(~attention_mask.bool().unsqueeze(-1), -1e9)
            max_p = masked_h.max(dim=1).values
            pooled = torch.cat([mean_p, max_p], dim=-1)
        else:
            mask = attention_mask.unsqueeze(-1).to(h.dtype)
            pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        pooled = pooled.float()
        with torch.autocast(pooled.device.type, enabled=False):
            logits = self.classifier(pooled)
            proj = self.proj_head(pooled) if self.use_proj_head and self.training else None
        return LogOutput(logits=logits, embeddings=pooled, projection=proj)
