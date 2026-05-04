from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


@dataclass
class LogADOutput:
    logits: torch.Tensor
    embeddings: torch.Tensor


class ModernBERTForLogAD(nn.Module):
    def __init__(self, model_path, num_labels, attn_implementation="sdpa", dtype=torch.bfloat16):
        super().__init__()
        config = AutoConfig.from_pretrained(model_path)
        self.encoder = AutoModel.from_pretrained(
            model_path, attn_implementation=attn_implementation, dtype=dtype,
        )
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).to(out.last_hidden_state.dtype)
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1)
        pooled = pooled.float()
        return LogADOutput(logits=self.classifier(pooled), embeddings=pooled)
