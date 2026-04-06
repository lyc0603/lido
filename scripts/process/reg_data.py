"""
Build regression dataset: redemption queue sensitivity to stETH discount.

Outputs: processed_data/reg_queue_discount.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parents[2]
PROCESSED = ROOT / "processed_data"


def load_hourly_price() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED / "steth_eth_hourly.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["date"] = df["timestamp"].dt.strftime("%Y-%m-%d")

    # Total stETH supply: market cap / price
    # steth_mcap is 0 for the very first rows; drop those
    df = df[df["steth_mcap"] > 0].copy()
    df["steth_outstanding"] = df["steth_mcap"] / df["steth_price"]

    # Daily aggregations
    daily_min = df.groupby("date").agg(
        min_steth_price=("steth_price", "min"),
        min_discount=("discount", "min"),
    )

    # End-of-day outstanding supply (last hourly observation)
    daily_outstanding = (
        df.sort_values("timestamp")
        .groupby("date")["steth_outstanding"]
        .last()
        .rename("steth_outstanding")
    )

    return daily_min.join(daily_outstanding).reset_index()


def load_queue_daily() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED / "queue_daily.csv")
    df["date"] = df["date"].astype(str)
    return df[["date", "steth_requested"]]


def build_reg_data() -> pd.DataFrame:
    price = load_hourly_price()
    queue = load_queue_daily()

    df = queue.merge(price, on="date", how="inner")

    # Redemption volume as proportion of outstanding supply
    # Use steth_requested: the daily decision to enter the queue (response to price signal)
    df["redemption_pct"] = df["steth_requested"] / df["steth_outstanding"]

    # Discount in percent
    df["min_discount_pct"] = df["min_discount"] * 100

    # Year for fixed effects
    df["year"] = pd.to_datetime(df["date"]).dt.year

    df = df.sort_values("date").reset_index(drop=True)

    cols = [
        "date",
        "year",
        "redemption_pct",
        "min_steth_price",
        "min_discount",
        "min_discount_pct",
        "steth_requested",
        "steth_outstanding",
    ]
    return df[cols]


if __name__ == "__main__":
    out = PROCESSED / "reg_queue_discount.csv"
    df = build_reg_data()
    df.to_csv(out, index=False)

    print(f"Saved {len(df):,} rows → {out}")
    print(df.describe().to_string())
