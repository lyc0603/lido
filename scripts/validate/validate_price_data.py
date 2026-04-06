"""
Visualize and validate stETH and ETH price data fetched from CoinMarketCap.
"""

import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns

from environ.constants import DATA_PATH as DATA_DIR, FIGURE_PATH as FIG_DIR

# Style

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 150, "figure.figsize": (12, 4)})

BLUE = "#2563EB"
ORANGE = "#EA580C"


# Loaders


def load(name: str) -> pd.DataFrame:
    path = DATA_DIR / f"{name}_price_raw.csv"
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# Validation


def validate(df: pd.DataFrame, name: str) -> bool:
    label = name.upper()
    ok = True

    # Basic info
    print(f"\n{'─'*56}")
    print(
        f"  {label}  ({len(df):,} rows,  {df['timestamp'].min()} → {df['timestamp'].max()})"
    )
    print(f"{'─'*56}")

    # 1. Duplicates
    dupes = df["timestamp"].duplicated().sum()
    if dupes:
        print(f"  ✗  Duplicate timestamps: {dupes}")
        ok = False
    else:
        print(f"  ✓  No duplicate timestamps")

    # 2. Missing hours (calendar gaps)
    full_range = pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq="h")
    missing = full_range.difference(df["timestamp"])
    if len(missing):
        print(f"  ✗  Missing hourly timestamps: {len(missing)}  (first: {missing[0]})")
        ok = False
    else:
        print(f"  ✓  No hourly gaps")

    # 3. Zero / negative price
    bad_price = (df["price_usd"] <= 0).sum()
    if bad_price:
        print(f"  ✗  Zero/negative price_usd: {bad_price} rows")
        ok = False
    else:
        print(f"  ✓  All prices positive")

    # 4. Zero market cap
    zero_mcap = (df["market_cap_usd"] == 0).sum()
    if zero_mcap:
        print(f"  ⚠  Zero market_cap_usd: {zero_mcap} rows")
    else:
        print(f"  ✓  All market caps non-zero")

    # 5. Volume spikes (> 5× 720-hour rolling median, ~30-day window)
    roll_med = df["volume_24h_usd"].rolling(720, min_periods=1).median()
    spikes = (df["volume_24h_usd"] > 5 * roll_med).sum()
    print(f"  {'⚠' if spikes else '✓'}  Volume spikes (>5× rolling median): {spikes}")

    # 6. Price range summary
    print(
        f"\n  Price (USD): min={df['price_usd'].min():,.4f}  "
        f"max={df['price_usd'].max():,.2f}  "
        f"latest={df['price_usd'].iloc[-1]:,.4f}"
    )

    return ok


def validate_ratio(eth: pd.DataFrame, steth: pd.DataFrame) -> None:
    print(f"\n{'─'*56}")
    print(f"  stETH / ETH  RATIO")
    print(f"{'─'*56}")

    merged = pd.merge(
        steth[["timestamp", "price_usd"]].rename(columns={"price_usd": "steth"}),
        eth[["timestamp", "price_usd"]].rename(columns={"price_usd": "eth"}),
        on="timestamp",
    )
    merged["ratio"] = merged["steth"] / merged["eth"]

    above_par = (merged["ratio"] > 1.0).sum()
    below_99 = (merged["ratio"] < 0.99).sum()
    min_ratio = merged["ratio"].min()
    min_ts = merged.loc[merged["ratio"].idxmin(), "timestamp"]

    print(f"  Ratio range: {min_ratio:.6f} → {merged['ratio'].max():.6f}")
    print(f"  Hours above parity (>1.000): {above_par}")
    print(f"  Hours below 0.990:           {below_99}")
    print(f"  Largest discount: {(1 - min_ratio)*100:.2f}%  at {min_ts}")
    print(f"  Latest ratio:     {merged['ratio'].iloc[-1]:.6f}")


# Plots


def _save(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


def plot_price(eth: pd.DataFrame, steth: pd.DataFrame) -> None:
    """Single unified price chart: ETH from 2015, stETH from its launch."""
    fig, ax = plt.subplots()

    ax.semilogy(
        eth["timestamp"], eth["price_usd"], color=BLUE, linewidth=0.5, label="ETH"
    )
    ax.semilogy(
        steth["timestamp"],
        steth["price_usd"],
        color=ORANGE,
        linewidth=0.5,
        label="stETH",
    )

    # Shade the pre-stETH period to make the difference in history visible
    ax.axvspan(
        eth["timestamp"].min(),
        steth["timestamp"].min(),
        alpha=0.06,
        color=BLUE,
        label="_nolegend_",
    )

    ax.set_title("ETH and stETH – price history (log scale)")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    fig.tight_layout()
    _save(fig, "price.png")


def plot_ratio(eth: pd.DataFrame, steth: pd.DataFrame) -> None:
    merged = pd.merge(
        steth[["timestamp", "price_usd"]].rename(columns={"price_usd": "steth"}),
        eth[["timestamp", "price_usd"]].rename(columns={"price_usd": "eth"}),
        on="timestamp",
    )
    merged["ratio"] = merged["steth"] / merged["eth"]

    fig, ax = plt.subplots()
    ax.plot(merged["timestamp"], merged["ratio"], color=ORANGE, linewidth=0.4)
    ax.axhline(1.0, color="black", linewidth=0.6, linestyle="--", label="Parity (1.0)")
    ax.set_title("stETH / ETH price ratio")
    ax.set_ylabel("Ratio")
    ax.legend()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    # Annotate deepest discount
    idx = merged["ratio"].idxmin()
    ax.annotate(
        f"  {merged.loc[idx,'ratio']:.3f}\n  {merged.loc[idx,'timestamp'].strftime('%b %Y')}",
        xy=(merged.loc[idx, "timestamp"], merged.loc[idx, "ratio"]),
        fontsize=7,
        color="red",
    )
    fig.tight_layout()
    _save(fig, "steth_eth_ratio.png")


def plot_volume(eth: pd.DataFrame, steth: pd.DataFrame) -> None:
    start = steth["timestamp"].min()
    eth_s = eth[eth["timestamp"] >= start].copy()
    steth = steth.copy()

    roll = 720  # 720 hours ≈ 30 days
    eth_s["vol_roll"] = eth_s["volume_24h_usd"].rolling(roll).mean()
    steth["vol_roll"] = steth["volume_24h_usd"].rolling(roll).mean()

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for ax, df, label, color in [
        (axes[0], eth_s, "ETH", BLUE),
        (axes[1], steth, "stETH", ORANGE),
    ]:
        ax.fill_between(
            df["timestamp"], df["volume_24h_usd"] / 1e9, alpha=0.3, color=color
        )
        ax.plot(
            df["timestamp"],
            df["vol_roll"] / 1e9,
            color=color,
            linewidth=1.0,
            label="30-day avg",
        )
        ax.set_ylabel("Volume (USD bn)")
        ax.set_title(f"{label} – 24h trading volume")
        ax.legend(fontsize=8)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.1f}B"))

    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    _save(fig, "volume.png")


def plot_market_cap(eth: pd.DataFrame, steth: pd.DataFrame) -> None:
    start = steth["timestamp"].min()
    eth_s = eth[(eth["timestamp"] >= start) & (eth["market_cap_usd"] > 0)]
    steth = steth[steth["market_cap_usd"] > 0]

    fig, ax = plt.subplots()
    ax.semilogy(
        eth_s["timestamp"],
        eth_s["market_cap_usd"] / 1e9,
        color=BLUE,
        linewidth=0.5,
        label="ETH",
    )
    ax.semilogy(
        steth["timestamp"],
        steth["market_cap_usd"] / 1e9,
        color=ORANGE,
        linewidth=0.5,
        label="stETH",
    )
    ax.set_title("Market capitalisation (log scale)")
    ax.set_ylabel("Market cap (USD bn)")
    ax.legend()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}B"))
    fig.tight_layout()
    _save(fig, "market_cap.png")


# Main


def main() -> None:
    eth = load("eth")
    steth = load("steth")

    # Validation report
    print("\n══════════════════════════════════════════════════════")
    print("  DATA VALIDATION REPORT")
    print("══════════════════════════════════════════════════════")

    eth_ok = validate(eth, "eth")
    steth_ok = validate(steth, "steth")
    validate_ratio(eth, steth)

    all_ok = eth_ok and steth_ok
    print(f"\n{'══'*28}")
    print(f"  Overall: {'✓ PASS' if all_ok else '✗ ISSUES FOUND'}")
    print(f"{'══'*28}\n")

    # Figures
    print("Saving figures …")
    plot_price(eth, steth)
    plot_ratio(eth, steth)
    plot_volume(eth, steth)
    plot_market_cap(eth, steth)
    print("Done.")

    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
