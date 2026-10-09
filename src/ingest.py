"""Download the AI4I 2020 dataset from UCI and store it as data/raw/ai4i2020.csv."""
import io
import urllib.request
import zipfile

import pandas as pd

from src.config import DATA_URL, RAW_PATH, RENAME


def download(force: bool = False) -> None:
    if RAW_PATH.exists() and not force:
        print(f"Raw data already at {RAW_PATH}")
        return
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {DATA_URL}")
    with urllib.request.urlopen(DATA_URL, timeout=60) as resp:
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        csv_name = next(n for n in zf.namelist() if n.endswith(".csv"))
        RAW_PATH.write_bytes(zf.read(csv_name))
    print(f"Saved {RAW_PATH}")


def load_raw() -> pd.DataFrame:
    """Load the raw CSV with snake_case column names."""
    return pd.read_csv(RAW_PATH, encoding="utf-8-sig").rename(columns=RENAME)


if __name__ == "__main__":
    download()
    df = load_raw()
    print(df.shape, f"failure rate = {df['machine_failure'].mean():.3%}")
