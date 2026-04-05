"""
Build a daily time-series of the Lido withdrawal queue state and merge it
with stETH/ETH price data.

Inputs  (from data/):
  withdrawal_requested.csv   – from fetch_lido_queue_events.py
  withdrawals_finalized.csv  – from fetch_lido_queue_events.py
  steth_eth_daily.csv        – from fetch_steth_price.py
  steth_eth_kaiko.csv        – optional, from fetch_steth_price.py

Output (in data/):
  queue_daily.csv            – daily queue snapshot
  queue_steth_merged.csv     – queue snapshot joined with stETH/ETH price

Daily queue columns produced
──────────────────────────────────────────────────────────────────────────────
  date
  requests_submitted      : new WithdrawalRequested events on this day
  steth_requested         : stETH requested this day (ETH-equivalent)
  cumulative_requests     : total requests ever submitted (≈ last request id)
  requests_finalized      : requests finalized this day
  eth_finalized           : ETH locked for finalization this day
  cumulative_finalized    : total finalized so far
  queue_length            : cumulative_requests − cumulative_finalized (unfinalized count)
  queue_steth             : rolling unfinalized stETH (requested − finalized stETH equivalent)
  finalization_rate_daily : requests_finalized / requests_submitted (0 when no submissions)

Usage:
  python scripts/build_queue_timeseries.py [--freq daily|hourly]
"""

import argparse
from pathlib import Path

import pandas as pd

from environ.constants import DATA_PATH as DATA_DIR, PROCESSED_DATA_PATH as PROCESSED_DIR


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_requested(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df["date"]      = df["timestamp"].dt.normalize()
    return df


def load_finalized(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df["date"]      = df["timestamp"].dt.normalize()
    return df


def load_price_daily(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"], utc=True)
    # Keep whichever optional columns are present
    base_cols   = ["date", "price_eth", "discount_pct"]
    extra_cols  = [c for c in ["volume_eth", "open", "high", "low", "close",
                                "steth_usd", "eth_usd"]
                   if c in df.columns]
    return df[base_cols + extra_cols].copy()


def load_price_kaiko(path: Path, freq: str = "daily") -> pd.DataFrame:
    """Aggregate Kaiko tick data to daily (or hourly) frequency."""
    df = pd.read_csv(path, parse_dates=["datetime"])
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    if freq == "daily":
        df["date"] = df["datetime"].dt.normalize()
        agg = df.groupby("date").agg(
            open         = ("open",  "first"),
            high         = ("high",  "max"),
            low          = ("low",   "min"),
            close        = ("close", "last"),
            volume_kaiko = ("volume","sum"),
            trades       = ("count", "sum"),
        ).reset_index()
        agg["price_eth"]   = agg["close"]
        agg["discount_pct"]= (1 - agg["close"]) * 100
        return agg
    else:   # hourly – return as-is with a consistent column name
        df = df.rename(columns={"datetime": "date", "volume": "volume_kaiko", "count": "trades"})
        df["price_eth"]    = df["close"]
        df["discount_pct"] = (1 - df["close"]) * 100
        return df


# ── Queue aggregate ────────────────────────────────────────────────────────────

def build_daily_queue(req: pd.DataFrame, fin: pd.DataFrame) -> pd.DataFrame:
    # Daily submissions
    req_daily = (
        req.groupby("date")
        .agg(
            requests_submitted = ("request_id",    "count"),
            steth_requested    = ("amount_steth",  "sum"),
            max_request_id     = ("request_id",    "max"),
        )
        .reset_index()
    )

    # Daily finalizations
    fin_daily = (
        fin.groupby("date")
        .agg(
            requests_finalized = ("requests_finalized", "sum"),
            eth_finalized      = ("amount_eth_locked",  "sum"),
        )
        .reset_index()
    )

    # Full date range
    all_dates = pd.date_range(
        start = min(req["date"].min(), fin["date"].min()),
        end   = max(req["date"].max(), fin["date"].max()),
        freq  = "D",
        tz    = "UTC",
    )
    df = pd.DataFrame({"date": all_dates})
    df = df.merge(req_daily, on="date", how="left")
    df = df.merge(fin_daily, on="date", how="left")
    df = df.fillna(0)

    # Cumulative counters
    df = df.sort_values("date").reset_index(drop=True)
    df["cumulative_requests"]  = df["requests_submitted"].cumsum()
    df["cumulative_finalized"] = df["requests_finalized"].cumsum()
    df["queue_length"]         = df["cumulative_requests"] - df["cumulative_finalized"]
    df["queue_steth"]          = (df["steth_requested"].cumsum()
                                  - df["eth_finalized"].cumsum())

    # Finalization rate: what fraction of today's queue was cleared today
    df["finalization_rate_daily"] = df.apply(
        lambda r: r["requests_finalized"] / r["requests_submitted"]
        if r["requests_submitted"] > 0 else float("nan"),
        axis=1,
    )

    return df


# ── Merge ─────────────────────────────────────────────────────────────────────

def merge_with_price(queue_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    price_cols = [c for c in price_df.columns if c != "date"]
    return queue_df.merge(price_df[["date"] + price_cols], on="date", how="left")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build queue time-series")
    parser.add_argument("--freq", choices=["daily", "hourly"], default="daily",
                        help="Output frequency (default: daily)")
    args = parser.parse_args()

    req_path   = DATA_DIR / "withdrawal_requested.csv"
    fin_path   = DATA_DIR / "withdrawals_finalized.csv"
    price_path = DATA_DIR / "steth_eth_daily.csv"
    kaiko_path = DATA_DIR / "steth_eth_kaiko.csv"

    for p, name in [(req_path, "withdrawal_requested.csv"),
                    (fin_path, "withdrawals_finalized.csv"),
                    (price_path, "steth_eth_daily.csv")]:
        if not p.exists():
            raise FileNotFoundError(
                f"Missing {name} – run the fetch scripts first.\n"
                f"  python scripts/fetch_lido_queue_events.py\n"
                f"  python scripts/fetch_steth_price.py"
            )

    print("Loading events …")
    req = load_requested(req_path)
    fin = load_finalized(fin_path)
    print(f"  {len(req):,} withdrawal requests, {len(fin):,} finalization batches")

    print("Building daily queue aggregates …")
    queue_df = build_daily_queue(req, fin)
    out_queue = PROCESSED_DIR / "queue_daily.csv"
    queue_df.to_csv(out_queue, index=False)
    print(f"  Saved {len(queue_df):,} daily rows → {out_queue}")

    # Price data: prefer Kaiko if available, fall back to CoinGecko
    if kaiko_path.exists():
        print("Loading Kaiko price data …")
        price_df = load_price_kaiko(kaiko_path, freq=args.freq)
    else:
        print("Loading CoinGecko price data (Kaiko not found) …")
        price_df = load_price_daily(price_path)

    print("Merging queue with price data …")
    merged = merge_with_price(queue_df, price_df)
    out_merged = PROCESSED_DIR / "queue_steth_merged.csv"
    merged.to_csv(out_merged, index=False)
    print(f"  Saved {len(merged):,} rows → {out_merged}")

    # Summary statistics
    print("\n── Summary ──────────────────────────────────────────────────────")
    print(f"  Date range       : {queue_df['date'].min().date()} → {queue_df['date'].max().date()}")
    print(f"  Total requests   : {int(queue_df['cumulative_requests'].max()):,}")
    print(f"  Total finalized  : {int(queue_df['cumulative_finalized'].max()):,}")
    print(f"  Current queue    : {int(queue_df['queue_length'].iloc[-1]):,}")
    if "discount_pct" in merged.columns and merged["discount_pct"].notna().any():
        valid = merged["discount_pct"].dropna()
        print(f"  Avg discount     : {valid.mean():.4f}%")
        print(f"  Max discount     : {valid.max():.4f}%")
        print(f"  Min discount     : {valid.min():.4f}%")
    print("─────────────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
