from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel


@dataclass
class LogOutput:
    logits: torch.Tensor
    embeddings: torch.Tensor


class AttentionPool(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states, attention_mask):
        scores = self.attn(hidden_states).squeeze(-1)
        scores = scores.masked_fill(~attention_mask.bool(), -1e9)
        weights = F.softmax(scores, dim=1).unsqueeze(-1)
        return (hidden_states * weights).sum(dim=1)


class ModernBertClassifier(nn.Module):
    def __init__(self, model_path, num_labels, pooling="mean",
                 attn_implementation=None, dtype=torch.bfloat16):
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
        if pooling == "attention":
            self.pool_attn = AttentionPool(config.hidden_size).to(dtype=dtype)
            cls_in = config.hidden_size
        elif pooling == "mean_max":
            cls_in = 2 * config.hidden_size
        else:
            cls_in = config.hidden_size
        self.classifier = nn.Linear(cls_in, num_labels)
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask, **kwargs):
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
        return LogOutput(logits=self.classifier(pooled), embeddings=pooled)
