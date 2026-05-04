from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent

STRUCTURED_PATH = {
    "BGL":         "data/BGL/BGL.log_structured.csv",
    "Liberty":     "data/Liberty/liberty2_structured.csv",
    "Thunderbird": "data/Thunderbird/Thunderbird.log_structured.csv",
}

DATASETS_TO_BUILD = ["BGL", "Liberty", "Thunderbird"]
WINDOW_SIZE  = 100
STEP_SIZE    = 100
TRAIN_RATIO  = 0.8
SEPARATOR    = " ;-; "
NORMAL_TOKEN = "-"


def session_label(window_labels: list[str]) -> str:
    counter = Counter(lbl for lbl in window_labels if lbl != NORMAL_TOKEN)
    return counter.most_common(1)[0][0] if counter else "normal"


def fixed_size_windows(df: pd.DataFrame, window_size: int, step_size: int) -> pd.DataFrame:
    contents, labels, lengths = [], [], []
    n = len(df)
    contents_arr = df["Content"].astype(str).values
    labels_arr = df["Label"].astype(str).values
    for start in range(0, n - window_size + 1, step_size):
        end = start + window_size
        contents.append(SEPARATOR.join(contents_arr[start:end]))
        labels.append(session_label(labels_arr[start:end].tolist()))
        lengths.append(window_size)
    return pd.DataFrame({"Content": contents, "Label": labels, "session_length": lengths})


def chronological_split(df: pd.DataFrame, train_ratio: float):
    cut = int(len(df) * train_ratio)
    return df.iloc[:cut].reset_index(drop=True), df.iloc[cut:].reset_index(drop=True)


def write_info(out_path: Path, df: pd.DataFrame) -> None:
    counts = df["Label"].value_counts().sort_values(ascending=False)
    n_normal = int(counts.get("normal", 0))
    info = [
        f"total_windows: {len(df)}",
        f"normal: {n_normal}",
        f"anomaly: {len(df) - n_normal}",
        f"unique_anomaly_types: {int((counts.index != 'normal').sum())}",
        "",
        "label distribution:",
    ]
    info.extend(f"  {lbl}: {cnt}" for lbl, cnt in counts.items())
    out_path.write_text("\n".join(info))


def process_dataset(dataset: str) -> None:
    structured_csv = ROOT / STRUCTURED_PATH[dataset]
    if not structured_csv.exists():
        raise FileNotFoundError(structured_csv)

    print(f"\n{dataset}: reading {structured_csv.name}")
    df = pd.read_csv(structured_csv, usecols=["Label", "Content"]).dropna(subset=["Content"]).reset_index(drop=True)
    print(f"  lines={len(df):,}")

    windowed = fixed_size_windows(df, WINDOW_SIZE, STEP_SIZE)
    train_df, test_df = chronological_split(windowed, TRAIN_RATIO)
    print(f"  windows={len(windowed):,} -> train={len(train_df):,} test={len(test_df):,}")

    out_dir = ROOT / "data" / dataset
    train_df.to_csv(out_dir / "train.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)
    write_info(out_dir / "train_info.txt", train_df)
    write_info(out_dir / "test_info.txt", test_df)
    print(f"  types train={len(set(train_df['Label']))} test={len(set(test_df['Label']))}")


def main() -> None:
    for d in DATASETS_TO_BUILD:
        process_dataset(d)


if __name__ == "__main__":
    main()
