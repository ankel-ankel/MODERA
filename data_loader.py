import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler


LOG_SEP = " ;-; "

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


class LogDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path).dropna(subset=["Content"]).reset_index(drop=True)
        self.labels = df["Label"].astype(str).values
        self.sequences = np.empty(len(df), dtype=object)
        for i, content in enumerate(df["Content"].values):
            self.sequences[i] = [mask_log(m) for m in str(content).split(LOG_SEP)]

        n_anom = int((self.labels != "normal").sum())
        n_types = len(np.unique(self.labels))
        print(f"[{csv_path}] n={len(df)} anom={n_anom} types={n_types}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def balanced_sampler(labels, target_ratio=0.3):
    is_anom = (labels != "normal").astype(np.int64)
    n_anom = int(is_anom.sum())
    n_norm = len(labels) - n_anom
    if n_anom == 0 or n_norm == 0:
        return WeightedRandomSampler(torch.ones(len(labels)), num_samples=len(labels), replacement=True)
    w_anom = target_ratio / n_anom
    w_norm = (1 - target_ratio) / n_norm
    weights = np.where(is_anom, w_anom, w_norm).astype(np.float32)
    return WeightedRandomSampler(
        torch.from_numpy(weights), num_samples=len(labels), replacement=True,
    )


@dataclass
class LogCollator:
    tokenizer: object
    label2id: dict
    max_token_len: int = 1024
    join_sep: str = " ;; "

    def __call__(self, batch):
        texts = [self.join_sep.join(logs) for logs, _ in batch]
        ids = [self.label2id.get(lbl, self.label2id.get("normal", 0)) for _, lbl in batch]
        enc = self.tokenizer(
            texts, return_tensors="pt", max_length=self.max_token_len,
            padding=True, truncation=True,
        )
        enc["labels"] = torch.tensor(ids, dtype=torch.long)
        return enc
