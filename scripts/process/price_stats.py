"""
Process raw stETH and ETH hourly price data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from environ.constants import DATA_PATH as DATA_DIR, PROCESSED_DATA_PATH as OUT_DIR

HOURS_PER_YEAR = 8_760  # 365 × 24
VOL_WINDOW_24H = 24  # 24-hour realized vol
VOL_WINDOW_7D = 24 * 7  # 7-day realized vol (168 hours)
# Anomaly detection: centered 96-hour window, flag only if > 50× the local median.
# A centered window avoids flagging organic ramp-ups or sustained stress events;
# the 50× threshold is calibrated to the confirmed ETH case (~113× the normal level).
ANOMALY_WINDOW = 96
ANOMALY_THRESHOLD = 50


# Load


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    eth = pd.read_csv(DATA_DIR / "eth_price_raw.csv", parse_dates=["timestamp"])
    steth = pd.read_csv(DATA_DIR / "steth_price_raw.csv", parse_dates=["timestamp"])
    eth = eth.sort_values("timestamp").reset_index(drop=True)
    steth = steth.sort_values("timestamp").reset_index(drop=True)
    return eth, steth


# Launch-day flat price fix


def drop_flat_launch_prices(df: pd.DataFrame, name: str = "") -> pd.DataFrame:
    """Drop leading rows where CMC forward-filled the launch price.

    At asset launch, CMC repeats a single price across multiple consecutive hours
    until real trade data arrives.  These stale prices produce spurious zero log
    returns and should be removed before computing any return-based statistics.
    """
    df = df.copy()
    prices = df["price_usd"].values
    cut = 0
    while cut < len(prices) - 1 and prices[cut] == prices[cut + 1]:
        cut += 1
    if cut > 0:
        print(
            f"  [{name}] Dropping {cut} launch-day flat-price rows "
            f"({df['timestamp'].iloc[0]} → {df['timestamp'].iloc[cut - 1]})"
        )
        df = df.iloc[cut:].reset_index(drop=True)
    else:
        print(f"  [{name}] No leading flat-price rows detected")
    return df


# ── Anomaly fix ───────────────────────────────────────────────────────────────


def fix_volume_anomalies(df: pd.DataFrame, name: str = "") -> pd.DataFrame:
    """Replace volume entries > ANOMALY_THRESHOLD × centered rolling median with NaN,
    then linearly interpolate.

    Uses a centered window so organic ramp-ups and sustained stress episodes are
    not mislabeled — only isolated transient spikes (like the confirmed ETH $4T case)
    are flagged.
    """
    df = df.copy()
    roll_med = (
        df["volume_24h_usd"]
        .rolling(ANOMALY_WINDOW, min_periods=ANOMALY_WINDOW // 4, center=True)
        .median()
    )
    bad = df["volume_24h_usd"] > ANOMALY_THRESHOLD * roll_med
    n_bad = bad.sum()
    if n_bad:
        print(f"  [{name}] Fixing {n_bad} anomalous volume entries:")
        for ts in df.loc[bad, "timestamp"]:
            orig = df.loc[df["timestamp"] == ts, "volume_24h_usd"].values[0]
            print(f"    {ts}  original={orig:,.0f}")
        df.loc[bad, "volume_24h_usd"] = float("nan")
        df["volume_24h_usd"] = df["volume_24h_usd"].interpolate(method="linear")
    else:
        print(f"  [{name}] No volume anomalies detected")
    return df


# Hourly panel


def build_hourly_panel(eth: pd.DataFrame, steth: pd.DataFrame) -> pd.DataFrame:
    """Merge on timestamp, compute discount, log returns, rolling volatility."""
    merged = pd.merge(
        eth.rename(
            columns={
                "price_usd": "eth_price",
                "volume_24h_usd": "eth_volume",
                "market_cap_usd": "eth_mcap",
            }
        ),
        steth.rename(
            columns={
                "price_usd": "steth_price",
                "volume_24h_usd": "steth_volume",
                "market_cap_usd": "steth_mcap",
            }
        ),
        on="timestamp",
    )

    # Discount: negative = stETH below ETH peg, positive = premium
    merged["discount"] = merged["steth_price"] / merged["eth_price"] - 1

    # Hourly log returns
    merged["eth_log_ret"] = np.log(merged["eth_price"] / merged["eth_price"].shift(1))
    merged["steth_log_ret"] = np.log(
        merged["steth_price"] / merged["steth_price"].shift(1)
    )

    # 24h rolling realized volatility (annualized)
    merged["eth_vol_24h"] = merged["eth_log_ret"].rolling(
        VOL_WINDOW_24H, min_periods=12
    ).std() * np.sqrt(HOURS_PER_YEAR)
    merged["steth_vol_24h"] = merged["steth_log_ret"].rolling(
        VOL_WINDOW_24H, min_periods=12
    ).std() * np.sqrt(HOURS_PER_YEAR)

    # 7-day rolling realized volatility (annualized) — smoother for visualization
    merged["eth_vol_7d"] = merged["eth_log_ret"].rolling(
        VOL_WINDOW_7D, min_periods=24
    ).std() * np.sqrt(HOURS_PER_YEAR)
    merged["steth_vol_7d"] = merged["steth_log_ret"].rolling(
        VOL_WINDOW_7D, min_periods=24
    ).std() * np.sqrt(HOURS_PER_YEAR)

    merged["timestamp"] = merged["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    cols = [
        "timestamp",
        "eth_price",
        "eth_volume",
        "eth_mcap",
        "steth_price",
        "steth_volume",
        "steth_mcap",
        "discount",
        "eth_log_ret",
        "steth_log_ret",
        "eth_vol_24h",
        "steth_vol_24h",
        "eth_vol_7d",
        "steth_vol_7d",
    ]
    return merged[cols]


# Daily panel


def build_daily_panel(hourly: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hourly panel to daily; keep end-of-day close values."""
    h = hourly.copy()
    h["timestamp"] = pd.to_datetime(h["timestamp"])
    h["date"] = h["timestamp"].dt.date

    daily = (
        h.groupby("date")
        .agg(
            eth_price_close=("eth_price", "last"),
            steth_price_close=("steth_price", "last"),
            # volume_24h_usd is a trailing-24h value updated every hour;
            # take the end-of-day reading as the daily volume
            eth_volume=("eth_volume", "last"),
            steth_volume=("steth_volume", "last"),
            eth_mcap_close=("eth_mcap", "last"),
            steth_mcap_close=("steth_mcap", "last"),
            discount_mean=("discount", "mean"),
            discount_close=("discount", "last"),
            discount_min=("discount", "min"),
            discount_max=("discount", "max"),
            steth_vol_7d=("steth_vol_7d", "last"),
            eth_vol_7d=("eth_vol_7d", "last"),
            n_obs=("discount", "count"),
        )
        .reset_index()
    )

    daily["is_discount"] = (daily["discount_mean"] < 0).astype(int)
    daily["is_premium"] = (daily["discount_mean"] > 0).astype(int)
    daily["date"] = daily["date"].astype(str)
    return daily


# Diagnostic plots


def plot_diagnostics(hourly: pd.DataFrame, daily: pd.DataFrame) -> None:
    h = hourly.copy()
    h["timestamp"] = pd.to_datetime(h["timestamp"])
    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"])

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("stETH / ETH Price Analysis – Diagnostic Overview", fontsize=13)

    # 1. Discount / premium time series (daily mean)
    ax = axes[0, 0]
    disc_pct = d["discount_mean"] * 100
    ax.fill_between(
        d["date"],
        disc_pct,
        0,
        where=disc_pct < 0,
        alpha=0.4,
        color="#EA580C",
        label="Discount",
    )
    ax.fill_between(
        d["date"],
        disc_pct,
        0,
        where=disc_pct > 0,
        alpha=0.4,
        color="#16A34A",
        label="Premium",
    )
    ax.plot(d["date"], disc_pct, linewidth=0.5, color="black")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("stETH Discount / Premium (% vs ETH)")
    ax.set_ylabel("%")
    ax.legend(fontsize=7)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())

    # 2. 7-day rolling annualized volatility
    ax = axes[0, 1]
    ax.plot(
        h["timestamp"],
        h["steth_vol_7d"] * 100,
        linewidth=0.5,
        color="#EA580C",
        label="stETH",
    )
    ax.plot(
        h["timestamp"],
        h["eth_vol_7d"] * 100,
        linewidth=0.5,
        color="#2563EB",
        label="ETH",
    )
    ax.set_title("7-Day Rolling Annualized Volatility (%)")
    ax.set_ylabel("%")
    ax.legend(fontsize=7)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())

    # 3. stETH 24h trading volume
    ax = axes[1, 0]
    ax.fill_between(d["date"], d["steth_volume"] / 1e6, alpha=0.6, color="#EA580C")
    ax.set_title("stETH 24h Trading Volume (USD mn)")
    ax.set_ylabel("USD mn")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())

    # 4. Discount distribution
    ax = axes[1, 1]
    disc = d["discount_mean"] * 100
    ax.hist(
        disc[disc < 0],
        bins=40,
        color="#EA580C",
        alpha=0.7,
        edgecolor="white",
        linewidth=0.3,
        label="Discount",
    )
    ax.hist(
        disc[disc > 0],
        bins=40,
        color="#16A34A",
        alpha=0.7,
        edgecolor="white",
        linewidth=0.3,
        label="Premium",
    )
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Distribution of Daily Mean Discount/Premium (%)")
    ax.set_xlabel("% vs ETH")
    ax.set_ylabel("Days")
    ax.legend(fontsize=7)

    fig.tight_layout()
    plt.show()


# Main


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading raw data …")
    eth, steth = load_raw()

    print("\nDropping launch-day flat prices …")
    eth = drop_flat_launch_prices(eth, "ETH")
    steth = drop_flat_launch_prices(steth, "stETH")

    print("\nFixing volume anomalies …")
    # Only ETH has a confirmed CMC data error ($4T volume on 2021-01-06 07-08h).
    # stETH volume spikes (e.g., 2022-10-26 pre-FTX contagion) are real market events.
    eth = fix_volume_anomalies(eth, "ETH")

    print("\nBuilding hourly panel …")
    hourly = build_hourly_panel(eth, steth)
    path = OUT_DIR / "steth_eth_hourly.csv"
    hourly.to_csv(path, index=False)
    print(f"  {len(hourly):,} rows → {path}")

    print("\nBuilding daily panel …")
    daily = build_daily_panel(hourly)
    path = OUT_DIR / "steth_eth_daily.csv"
    daily.to_csv(path, index=False)
    print(f"  {len(daily):,} days → {path}")

    disc = daily[daily["is_discount"] == 1]["discount_mean"] * 100
    prem = daily[daily["is_premium"] == 1]["discount_mean"] * 100
    print(
        f"\n  Discount days : {len(disc):,} / {len(daily):,}  ({len(disc)/len(daily)*100:.1f}%)"
    )
    print(
        f"  Premium days  : {len(prem):,} / {len(daily):,}  ({len(prem)/len(daily)*100:.1f}%)"
    )
    print(f"  Avg discount  : {disc.mean():.4f}%   Median: {disc.median():.4f}%")
    print(f"  Avg premium   : {prem.mean():.4f}%   Median: {prem.median():.4f}%")

    print("\nRendering diagnostic plots …")
    plot_diagnostics(hourly, daily)

    print("Done.")


if __name__ == "__main__":
    main()
