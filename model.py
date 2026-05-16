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
                 attn_implementation="sdpa", dtype=torch.bfloat16):
        super().__init__()
        config = AutoConfig.from_pretrained(model_path)
        self.encoder = AutoModel.from_pretrained(
            model_path, attn_implementation=attn_implementation, dtype=dtype,
        )
        self.pooling = pooling
        if pooling == "attention":
            self.pool_attn = AttentionPool(config.hidden_size)
        elif pooling == "mean_max":
            self.classifier = nn.Linear(2 * config.hidden_size, num_labels)
            self.num_labels = num_labels
            return
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
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
