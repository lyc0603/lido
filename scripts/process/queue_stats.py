"""
Process Lido withdrawal queue raw data into analysis-ready CSVs.
Also renders quick diagnostic figures via matplotlib (not saved).

Outputs → processed_data/:
  queue_daily.csv      – daily queue snapshot (length, stETH, flows)
  queue_requests.csv   – request-level panel with wait times
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from environ.constants import DATA_PATH as DATA_DIR, PROCESSED_DATA_PATH as OUT_DIR


# ── Load raw data ─────────────────────────────────────────────────────────────


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    req = pd.read_csv(DATA_DIR / "withdrawal_requested.csv")
    fin = pd.read_csv(DATA_DIR / "withdrawals_finalized.csv")
    req["submit_date"] = pd.to_datetime(req["timestamp"], unit="s", utc=True).dt.date
    fin["finalize_date"] = pd.to_datetime(fin["timestamp"], unit="s", utc=True).dt.date
    return req, fin


# ── Request-level panel with wait times ──────────────────────────────────────


def build_request_panel(req: pd.DataFrame, fin: pd.DataFrame) -> pd.DataFrame:
    """
    Join each request to its finalization batch.
    Wait time = finalization timestamp − submission timestamp.
    Unfinalized requests have NaN for all finalization columns.
    """
    # Expand finalization ranges into a request_id → finalization mapping
    fin_map_rows = []
    for _, row in fin.iterrows():
        for rid in range(int(row["from_request_id"]), int(row["to_request_id"]) + 1):
            fin_map_rows.append(
                {
                    "request_id": rid,
                    "fin_timestamp": row["timestamp"],
                    "fin_date": row["finalize_date"],
                    "max_share_rate": row["max_share_rate"],
                }
            )
    fin_map = pd.DataFrame(fin_map_rows)

    panel = req[
        [
            "request_id",
            "amount_steth",
            "amount_shares",
            "timestamp",
            "submit_date",
            "block_number",
        ]
    ].merge(fin_map, on="request_id", how="left")

    panel["is_finalized"] = panel["fin_timestamp"].notna().astype(int)  # 1/0 for Stata
    panel["wait_seconds"] = panel["fin_timestamp"] - panel["timestamp"]
    panel["wait_days"] = panel["wait_seconds"] / 86_400

    # Keep columns Stata will use
    out = panel[
        [
            "request_id",
            "submit_date",
            "timestamp",
            "amount_steth",
            "amount_shares",
            "is_finalized",
            "fin_date",
            "fin_timestamp",
            "wait_seconds",
            "wait_days",
            "max_share_rate",
        ]
    ].copy()
    out["submit_year"] = pd.to_datetime(out["submit_date"]).dt.year
    return out.sort_values("request_id").reset_index(drop=True)


# ── Daily queue snapshot ──────────────────────────────────────────────────────


def build_queue_daily(req: pd.DataFrame, fin: pd.DataFrame) -> pd.DataFrame:
    """
    For every calendar date since queue launch, compute:
      - n_submitted / steth_submitted  : new requests on that day
      - n_finalized / steth_finalized  : requests cleared on that day
      - queue_length / queue_steth     : end-of-day queue stock (cumulative)
    """
    date_range = pd.date_range(
        start=pd.Timestamp("2023-05-15"),
        end=pd.Timestamp(req["submit_date"].max()),
        freq="D",
    )

    # Daily flows
    req_daily = (
        req.groupby("submit_date")
        .agg(
            n_submitted=("request_id", "count"), steth_submitted=("amount_steth", "sum")
        )
        .reset_index()
        .rename(columns={"submit_date": "date"})
    )
    req_daily["date"] = pd.to_datetime(req_daily["date"])

    fin_daily = (
        fin.groupby("finalize_date")
        .agg(
            n_finalized=("requests_finalized", "sum"),
            steth_finalized=("amount_eth_locked", "sum"),
        )
        .reset_index()
        .rename(columns={"finalize_date": "date"})
    )
    fin_daily["date"] = pd.to_datetime(fin_daily["date"])

    daily = (
        pd.DataFrame({"date": date_range})
        .merge(req_daily, on="date", how="left")
        .merge(fin_daily, on="date", how="left")
        .fillna(0)
    )

    daily["queue_length"] = (daily["n_submitted"] - daily["n_finalized"]).cumsum()
    daily["queue_steth"] = (
        daily["steth_submitted"] - daily["steth_finalized"]
    ).cumsum()
    daily["date"] = daily["date"].dt.strftime("%Y-%m-%d")

    cols = [
        "date",
        "n_submitted",
        "steth_submitted",
        "n_finalized",
        "steth_finalized",
        "queue_length",
        "queue_steth",
    ]
    return daily[cols]


# ── Quick diagnostic plots (temporary, not saved) ────────────────────────────


def plot_diagnostics(daily: pd.DataFrame, panel: pd.DataFrame) -> None:
    dates = pd.to_datetime(daily["date"])
    finalized = panel[panel["is_finalized"]]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Lido Withdrawal Queue – Diagnostic Overview", fontsize=13)

    # 1. Queue length over time
    ax = axes[0, 0]
    ax.plot(dates, daily["queue_length"], linewidth=0.9, color="#2563EB")
    ax.set_title("Queue Length (# requests)")
    ax.set_ylabel("Requests")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())

    # 2. Queue stETH over time
    ax = axes[0, 1]
    ax.plot(dates, daily["queue_steth"] / 1e3, linewidth=0.9, color="#EA580C")
    ax.set_title("Queue stETH (thousands)")
    ax.set_ylabel("stETH (k)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())

    # 3. Wait time distribution (finalized only)
    ax = axes[1, 0]
    ax.hist(
        finalized["wait_days"],
        bins=60,
        color="#2563EB",
        edgecolor="white",
        linewidth=0.3,
    )
    ax.set_title("Wait Time Distribution (finalized requests)")
    ax.set_xlabel("Days")
    ax.set_ylabel("Count")

    # 4. Monthly median wait time
    ax = axes[1, 1]
    panel2 = finalized.copy()
    panel2["month"] = pd.to_datetime(panel2["submit_date"]).dt.to_period("M")
    monthly = panel2.groupby("month")["wait_days"].median().reset_index()
    monthly["month_dt"] = monthly["month"].dt.to_timestamp()
    ax.plot(monthly["month_dt"], monthly["wait_days"], linewidth=0.9, color="#EA580C")
    ax.set_title("Monthly Median Wait Time")
    ax.set_xlabel("Submission Month")
    ax.set_ylabel("Days")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())

    fig.tight_layout()
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    req, fin = load_raw()

    print("Building request panel …")
    panel = build_request_panel(req, fin)
    path = OUT_DIR / "queue_requests.csv"
    panel.to_csv(path, index=False)
    print(f"  {len(panel):,} requests → {path}")
    finalized = panel["is_finalized"].sum()
    print(f"  Finalized: {finalized:,}  |  Pending: {len(panel)-finalized:,}")
    print(
        f"  Wait time (finalized): "
        f"mean={panel['wait_days'].mean():.1f}d  "
        f"median={panel['wait_days'].median():.1f}d  "
        f"max={panel['wait_days'].max():.1f}d"
    )

    print("\nBuilding daily queue snapshot …")
    daily = build_queue_daily(req, fin)
    path = OUT_DIR / "queue_daily.csv"
    daily.to_csv(path, index=False)
    print(f"  {len(daily):,} days → {path}")
    print(f"  Queue length (latest): {daily['queue_length'].iloc[-1]:,.0f} requests")
    print(f"  Queue stETH  (latest): {daily['queue_steth'].iloc[-1]:,.2f} stETH")

    print("\nRendering diagnostic plots …")
    plot_diagnostics(daily, panel)

    print("Done.")


if __name__ == "__main__":
    main()
