"""
Fetch stETH and ETH full-history daily price / market-cap / volume from CoinMarketCap.

Outputs (raw data, one file per asset):
  data/steth_price_raw.csv   – stETH (CMC id 8085)
  data/eth_price_raw.csv     – ETH   (CMC id 1027)

Each file columns:
  date, price_usd, volume_24h_usd, market_cap_usd

Usage:
  python scripts/fetch_steth_price.py

Requires:
  COINMARKETCAP_API_KEY in .env  (Hobbyist plan or above for historical data).
"""

import csv
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import requests

from environ.constants import DATA_PATH as DATA_DIR
from environ.settings import COINMARKETCAP_API_KEY

# ── Constants ─────────────────────────────────────────────────────────────────

CMC_URL = "https://pro-api.coinmarketcap.com/v2/cryptocurrency/quotes/historical"

# CMC asset IDs with the earliest date each asset appears in CMC data
CMC_ASSETS = {
    # time_start is exclusive; set one day before the desired first data point.
    # ETH:   first data point on CMC is 2015-08-08; use 2015-08-06 as time_start
    # stETH: earliest API data is 2020-12-24 (website shows 2020-12-23 in local TZ)
    "eth":   {"id": 1027, "start": date(2015, 8, 6)},
    "steth": {"id": 8085, "start": date(2020, 12, 23)},
}

# Stay well within CMC's per-call data-point limits
CHUNK_DAYS = 365

CSV_FIELDS = ["date", "price_usd", "volume_24h_usd", "market_cap_usd"]


# ── Helpers ───────────────────────────────────────────────────────────────────


def _isoformat(d: date) -> str:
    return datetime(d.year, d.month, d.day, tzinfo=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _extract_quotes(payload: dict, asset_id: int) -> list:
    """
    CMC v2 /quotes/historical returns data keyed by string ID, integer ID,
    or (for single-asset calls on some plan tiers) as the data object directly.
    Try each layout in order.
    """
    data = payload.get("data", {})
    # Layout 1: {"data": {"1027": {"quotes": [...]}}}
    if str(asset_id) in data:
        return data[str(asset_id)]["quotes"]
    # Layout 2: {"data": {1027: {"quotes": [...]}}}
    if asset_id in data:
        return data[asset_id]["quotes"]
    # Layout 3: {"data": {"quotes": [...]}}  (single-asset shorthand)
    if "quotes" in data:
        return data["quotes"]
    # Layout 4: {"data": [{"id": 1027, "quotes": [...]}]}
    if isinstance(data, list):
        for item in data:
            if item.get("id") == asset_id:
                return item["quotes"]
    raise KeyError(
        f"Cannot locate quotes for id={asset_id} in response. "
        f"Top-level keys: {list(data.keys()) if isinstance(data, dict) else type(data)}"
    )


# ── Fetch ─────────────────────────────────────────────────────────────────────


def fetch_quotes(asset_id: int, start: date, end: date, api_key: str) -> list[dict]:
    """
    Fetch daily quotes for one asset over the full [start, end] range.
    Splits into CHUNK_DAYS windows; retries on transient failures.
    """
    rows: list[dict] = []
    chunk_start = start

    while chunk_start <= end:
        chunk_end = min(chunk_start + timedelta(days=CHUNK_DAYS - 1), end)
        params = {
            "id": asset_id,
            "time_start": _isoformat(chunk_start),
            "time_end": _isoformat(chunk_end),
            "interval": "daily",
            "convert": "USD",
        }
        headers = {"X-CMC_PRO_API_KEY": api_key, "Accept": "application/json"}

        for attempt in range(3):
            try:
                resp = requests.get(CMC_URL, params=params, headers=headers, timeout=30)
                resp.raise_for_status()
                payload = resp.json()

                status = payload.get("status", {})
                if status.get("error_code", 0) != 0:
                    raise RuntimeError(status.get("error_message", "CMC API error"))

                quotes = _extract_quotes(payload, asset_id)
                for q in quotes:
                    usd = q["quote"]["USD"]
                    rows.append(
                        {
                            "date": q["timestamp"][:10],
                            "price_usd": usd["price"],
                            "volume_24h_usd": usd["volume_24h"],
                            "market_cap_usd": usd["market_cap"],
                        }
                    )
                print(f"    {chunk_start} → {chunk_end}: {len(quotes)} points")
                break

            except Exception as exc:
                if attempt < 2:
                    wait = 5.0 * (attempt + 1)
                    print(
                        f"    Attempt {attempt+1}/3 failed: {exc}  (retry in {wait}s)"
                    )
                    time.sleep(wait)
                else:
                    print(f"    ERROR: giving up on {chunk_start}→{chunk_end}: {exc}")
                    sys.exit(1)

        # Advance to chunk_end so the next chunk's time_start (exclusive) picks up
        # from chunk_end+1 with no boundary gap. Stop if we just finished the last chunk.
        if chunk_end >= end:
            break
        chunk_start = chunk_end

    # Deduplicate on date (the overlapping time_start produces no duplicate in practice,
    # but guard anyway in case CMC returns the boundary day twice)
    seen: dict[str, dict] = {}
    for r in rows:
        seen[r["date"]] = r
    return sorted(seen.values(), key=lambda r: r["date"])


# ── Writer ────────────────────────────────────────────────────────────────────


def save_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved {len(rows):,} rows → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    if not COINMARKETCAP_API_KEY:
        print("ERROR: COINMARKETCAP_API_KEY is not set in .env")
        sys.exit(1)

    today = date.today()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for asset, cfg in CMC_ASSETS.items():
        start = cfg["start"]
        print(f"=== {asset.upper()} (id={cfg['id']})  {start} → {today} ===")
        rows = fetch_quotes(cfg["id"], start, today, COINMARKETCAP_API_KEY)
        out = DATA_DIR / f"{asset}_price_raw.csv"
        save_csv(rows, out)
        print()

    print("Done.")


if __name__ == "__main__":
    main()
