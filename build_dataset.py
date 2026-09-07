from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

# đường dẫn file log đã tách cột
STRUCTURED_PATH = {
    "BGL":            "data/BGL/BGL.log_structured.csv",
    "HDFS_v1":        "data/HDFS_v1/HDFS.log_structured.csv",
    "Liberty":        "data/Liberty/liberty2_structured.csv",
    "Liberty_random": "data/Liberty/liberty2_structured.csv",
    "Thunderbird":    "data/Thunderbird/Thunderbird.log_structured.csv",
}

RANDOM_SPLIT = {"Liberty_random"}
RANDOM_SPLIT_SEED = 42

DATASETS_TO_BUILD = ["BGL", "Liberty", "Thunderbird"]
# cắt cửa sổ và tỉ lệ chia, đổi ở đây
WINDOW_SIZE  = 100
STEP_SIZE    = 100
TRAIN_RATIO  = 0.7
VAL_RATIO    = 0.1
TEST_RATIO   = 0.2
SEPARATOR    = " ;-; "
NORMAL_TOKEN = "-"


def session_label(window_labels: list[str]) -> str:
    return "anomaly" if any(lbl != NORMAL_TOKEN for lbl in window_labels) else "normal"


# cắt cửa sổ 100 dòng, không chồng lấn
def fixed_size_windows(df: pd.DataFrame, window_size: int, step_size: int) -> pd.DataFrame:
    contents, labels = [], []
    n = len(df)
    contents_arr = df["Content"].astype(str).values
    labels_arr = df["Label"].astype(str).values
    for start in range(0, n - window_size + 1, step_size):
        end = start + window_size
        contents.append(SEPARATOR.join(contents_arr[start:end]))
        labels.append(session_label(labels_arr[start:end].tolist()))
    return pd.DataFrame({"Content": contents, "Label": labels})


def chronological_split(df: pd.DataFrame, train_ratio: float, val_ratio: float):
    n = len(df)
    cut_train = int(n * train_ratio)
    cut_val = int(n * (train_ratio + val_ratio))
    return (
        df.iloc[:cut_train].reset_index(drop=True),
        df.iloc[cut_train:cut_val].reset_index(drop=True),
        df.iloc[cut_val:].reset_index(drop=True),
    )


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
    if dataset in RANDOM_SPLIT:
        rng = np.random.default_rng(RANDOM_SPLIT_SEED)
        windowed = windowed.iloc[rng.permutation(len(windowed))].reset_index(drop=True)
    train_df, val_df, test_df = chronological_split(windowed, TRAIN_RATIO, VAL_RATIO)
    print(f"  windows={len(windowed):,} -> train={len(train_df):,} val={len(val_df):,} test={len(test_df):,}")

    out_dir = ROOT / "data" / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)
    write_info(out_dir / "train_info.txt", train_df)
    write_info(out_dir / "val_info.txt", val_df)
    write_info(out_dir / "test_info.txt", test_df)
    print(f"  types train={len(set(train_df['Label']))} val={len(set(val_df['Label']))} test={len(set(test_df['Label']))}")


def main() -> None:
    for d in DATASETS_TO_BUILD:
        process_dataset(d)


if __name__ == "__main__":
    main()
