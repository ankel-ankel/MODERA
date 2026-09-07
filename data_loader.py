import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler


LOG_SEP = " ;-; "
JOIN_SEP = " ;; "

# regex thay giá trị biến bằng placeholder
_PATTERNS = [
    r"True", r"true", r"False", r"false",
    r"\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen"
    r"|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty"
    r"|sixty|seventy|eighty|ninety|hundred|thousand|million|billion)\b",
    r"\b(Mon|Monday|Tue|Tuesday|Wed|Wednesday|Thu|Thursday|Fri|Friday|Sat|Saturday|Sun|Sunday)\b",
    r"\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?"
    r"|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(\d{1,2})\s+\b",
    r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(:\d{1,5})?",
    r"([0-9A-Fa-f]{2}:){11}[0-9A-Fa-f]{2}",
    r"([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}",
    r"[a-zA-Z0-9]*[:\.]*([/\\]+[^/\\\s\[\]]+)+[/\\]*",
    r"\b[0-9a-fA-F]{8}\b",
    r"\b[0-9a-fA-F]{10}\b",
    r"(\w+[\w\.]*)@(\w+[\w\.]*)\-(\w+[\w\.]*)",
    r"(\w+[\w\.]*)@(\w+[\w\.]*)",
    r"[a-zA-Z\.\:\-\_]*\d[a-zA-Z0-9\.\:\-\_]*",
]
_COMBINED = re.compile("|".join(_PATTERNS))


def mask_log(text):
    text = re.sub(r"\.{3,}", ".. ", text)
    return _COMBINED.sub("<*>", text)


# đọc csv, mask rồi tokenize sẵn
class LogDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_token_len, join_sep=JOIN_SEP):
        df = pd.read_csv(csv_path).dropna(subset=["Content"]).reset_index(drop=True)
        self.labels = df["Label"].astype(str).values

        texts = [
            join_sep.join(mask_log(m) for m in str(content).split(LOG_SEP))
            for content in df["Content"].values
        ]

        enc = tokenizer(texts, max_length=max_token_len, truncation=True, padding=False)
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]

        n_anom = int((self.labels != "normal").sum())
        n_types = len(np.unique(self.labels))
        print(f"[{csv_path}] n={len(df)} anom={n_anom} types={n_types}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "label": self.labels[idx],
        }


# lấy mẫu lại cho cân bằng lớp
def balanced_sampler(labels, target_ratio=0.4, generator=None):
    is_anom = (labels != "normal").astype(np.int64)
    n_anom = int(is_anom.sum())
    n_norm = len(labels) - n_anom
    if n_anom == 0 or n_norm == 0:
        return WeightedRandomSampler(
            torch.ones(len(labels)), num_samples=len(labels), replacement=True,
            generator=generator,
        )
    target_anom = target_ratio if n_anom < n_norm else 1.0 - target_ratio
    w_anom = target_anom / n_anom
    w_norm = (1 - target_anom) / n_norm
    weights = np.where(is_anom, w_anom, w_norm).astype(np.float32)
    return WeightedRandomSampler(
        torch.from_numpy(weights), num_samples=len(labels), replacement=True,
        generator=generator,
    )


@dataclass
# pad theo batch, đổi nhãn chữ sang số
class LogCollator:
    tokenizer: object
    label2id: dict

    def __call__(self, batch):
        unknown = {b["label"] for b in batch} - set(self.label2id)
        if unknown:
            raise KeyError(f"labels not in label2id (would be silently misclassified): {sorted(unknown)}")
        ids = [self.label2id[b["label"]] for b in batch]
        padded = self.tokenizer.pad(
            {"input_ids": [b["input_ids"] for b in batch],
             "attention_mask": [b["attention_mask"] for b in batch]},
            return_tensors="pt",
        )
        padded["labels"] = torch.tensor(ids, dtype=torch.long)
        return padded
