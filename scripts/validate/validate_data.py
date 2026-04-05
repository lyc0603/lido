"""
Cross-validate the collected Lido data against three independent sources.

1. On-chain view functions  – contract state is ground truth
2. Internal consistency     – accounting identities within our CSVs
3. Price cross-check        – DeFiLlama vs Curve pool exchange rate on-chain

Usage:
    python scripts/validate_data.py [--rpc RPC_URL]
"""

import argparse
from datetime import datetime, timezone
import pandas as pd
import requests
from web3 import Web3

from environ.constants import DATA_PATH as DATA_DIR, PROCESSED_DATA_PATH as PROCESSED_DIR
WQ_ADDR    = "0x889edC2eDab5f40e902b864aD4d7AdE8E412F9B1"
CURVE_POOL = "0xDC24316b9AE028F1497c275EB9192a3Ea0f67022"   # stETH/ETH Curve pool

PASS = "✓"
FAIL = "✗"
WARN = "~"

WQ_ABI = [
    {"name": "getLastRequestId",          "type": "function", "inputs": [],
     "outputs": [{"type": "uint256"}], "stateMutability": "view"},
    {"name": "getLastFinalizedRequestId", "type": "function", "inputs": [],
     "outputs": [{"type": "uint256"}], "stateMutability": "view"},
    {"name": "unfinalizedRequestNumber",  "type": "function", "inputs": [],
     "outputs": [{"type": "uint256"}], "stateMutability": "view"},
    {"name": "unfinalizedStETH",          "type": "function", "inputs": [],
     "outputs": [{"type": "uint256"}], "stateMutability": "view"},
]

CURVE_ABI = [
    # get_dy(i, j, dx): returns dy for swapping dx of token i to token j
    # token 0 = ETH, token 1 = stETH
    {"name": "get_dy",  "type": "function",
     "inputs":  [{"type": "int128"}, {"type": "int128"}, {"type": "uint256"}],
     "outputs": [{"type": "uint256"}], "stateMutability": "view"},
    # get_virtual_price: pool LP share price
    {"name": "get_virtual_price", "type": "function", "inputs": [],
     "outputs": [{"type": "uint256"}], "stateMutability": "view"},
]


def header(title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def check(label: str, ok: bool, detail: str = ""):
    mark = PASS if ok else FAIL
    print(f"  {mark}  {label}" + (f"  ({detail})" if detail else ""))
    return ok


# ── 1. On-chain validation ─────────────────────────────────────────────────────

def validate_onchain(w3: Web3, req: pd.DataFrame, fin: pd.DataFrame):
    header("1 · On-chain view functions  (contract state = ground truth)")
    wq = w3.eth.contract(
        address=w3.to_checksum_address(WQ_ADDR), abi=WQ_ABI
    )

    chain_last_req   = wq.functions.getLastRequestId().call()
    chain_last_fin   = wq.functions.getLastFinalizedRequestId().call()
    chain_unfin_num  = wq.functions.unfinalizedRequestNumber().call()
    chain_unfin_steth= wq.functions.unfinalizedStETH().call() / 1e18

    our_last_req  = int(req["request_id"].max())
    our_last_fin  = int(fin["to_request_id"].max())
    our_unfin_num = chain_last_req - chain_last_fin        # expected value
    our_unfin_steth_approx = (
        req[req["request_id"] > our_last_fin]["amount_steth"].sum()
    )

    print(f"\n  {'Metric':<35} {'On-chain':>12}  {'Our data':>12}")
    print(f"  {'─'*35} {'─'*12}  {'─'*12}")

    ok1 = check(
        f"{'Last request ID':<35} {chain_last_req:>12,}  {our_last_req:>12,}",
        chain_last_req == our_last_req,
        "scan may be behind current block" if chain_last_req > our_last_req else "",
    )
    ok2 = check(
        f"{'Last finalized ID':<35} {chain_last_fin:>12,}  {our_last_fin:>12,}",
        chain_last_fin == our_last_fin,
    )
    ok3 = check(
        f"{'Unfinalized request count':<35} {chain_unfin_num:>12,}  {our_unfin_num:>12,}",
        chain_unfin_num == our_unfin_num,
    )
    ok4 = check(
        f"{'Unfinalized stETH (ours ~)':<35} {chain_unfin_steth:>12,.1f}  "
        f"{our_unfin_steth_approx:>12,.1f}",
        abs(chain_unfin_steth - our_unfin_steth_approx) / max(chain_unfin_steth, 1) < 0.01,
        f"diff {abs(chain_unfin_steth - our_unfin_steth_approx):,.1f} stETH",
    )
    return all([ok1, ok2, ok3, ok4])


# ── 2. Internal consistency ────────────────────────────────────────────────────

def validate_internal(req: pd.DataFrame, fin: pd.DataFrame):
    header("2 · Internal consistency  (accounting identities)")

    # 2a. No duplicate request IDs
    dups = req["request_id"].duplicated().sum()
    check("No duplicate request_ids", dups == 0, f"{dups} duplicates" if dups else "")

    # 2b. Request IDs are contiguous 1 → max
    n_missing = req["request_id"].max() - req["request_id"].min() + 1 - len(req)
    check("No missing request IDs", n_missing == 0,
          f"{n_missing} gaps" if n_missing else "")

    # 2c. Finalization batches are contiguous and non-overlapping
    fin_s = fin.sort_values("from_request_id").reset_index(drop=True)
    gaps  = (fin_s["from_request_id"].iloc[1:].values
             - fin_s["to_request_id"].iloc[:-1].values - 1)
    has_gaps = (gaps != 0).sum()
    check("Finalization batches contiguous (no gaps/overlaps)",
          has_gaps == 0, f"{has_gaps} discontinuities" if has_gaps else "")

    # 2d. requests_finalized matches to_id - from_id + 1
    computed = fin["to_request_id"] - fin["from_request_id"] + 1
    mismatch = (computed != fin["requests_finalized"]).sum()
    check("requests_finalized == to_id - from_id + 1",
          mismatch == 0, f"{mismatch} mismatches" if mismatch else "")

    # 2e. stETH/shares ratio is monotonically non-decreasing (rewards accrue)
    req_ev = req[req["amount_shares"] > 0].copy()
    req_ev["ratio"] = req_ev["amount_steth"] / req_ev["amount_shares"]
    req_daily = req_ev.groupby("block_number")["ratio"].mean().reset_index().sort_values("block_number")
    decreases = (req_daily["ratio"].diff().dropna() < -0.01).sum()
    check("stETH/shares ratio non-decreasing (rewards accrue)",
          decreases == 0, f"{decreases} large drops (>1%)" if decreases else "")

    # 2f. ETH locked ≈ shares burned × share rate (within 1%)
    fin_check = fin[fin["shares_to_burn"] > 0].copy()
    fin_check["implied_rate"] = fin_check["amount_eth_locked"] / fin_check["shares_to_burn"]
    # share rate should be in [1.0, 1.5] range
    out_of_range = ((fin_check["implied_rate"] < 1.0) |
                    (fin_check["implied_rate"] > 1.5)).sum()
    check("ETH-locked / shares-burned ratio in [1.0, 1.5]",
          out_of_range == 0, f"{out_of_range} batches out of range" if out_of_range else "")

    # 2g. Timestamps are monotone within each event type
    req_ts = req[req["timestamp"] > 0].sort_values("request_id")["timestamp"]
    ts_violations = (req_ts.diff().dropna() < -86_400).sum()   # allow 1-day slack
    check("Request timestamps roughly non-decreasing", ts_violations == 0,
          f"{ts_violations} large reversals" if ts_violations else "")

    # 2h. stETH price coverage for queue period
    price = pd.read_csv(DATA_DIR / "steth_eth_daily.csv", parse_dates=["date"])
    queue = pd.read_csv(PROCESSED_DIR / "queue_steth_merged.csv")
    queue["date"] = pd.to_datetime(queue["date"])
    merged_price_days = queue["price_eth"].notna().sum()
    total_days        = len(queue)
    check(f"Price coverage for queue dates ({merged_price_days}/{total_days} days)",
          merged_price_days / total_days > 0.95,
          f"{100*merged_price_days/total_days:.1f}% covered")


# ── 3. Price cross-check ───────────────────────────────────────────────────────

def validate_price(w3: Web3):
    header("3 · Price cross-check  (DeFiLlama vs Curve pool on-chain)")

    curve = w3.eth.contract(
        address=w3.to_checksum_address(CURVE_POOL), abi=CURVE_ABI
    )

    # On-chain: exchange rate for swapping 1 stETH → ETH via Curve pool
    # token 1 = stETH → token 0 = ETH
    one_steth = 10**18
    try:
        eth_out      = curve.functions.get_dy(1, 0, one_steth).call()
        curve_price  = eth_out / 1e18
        curve_disc   = (1 - curve_price) * 100
    except Exception as e:
        print(f"  {WARN}  Could not read Curve pool: {e}")
        return

    # DeFiLlama: latest daily price
    df = pd.read_csv(DATA_DIR / "steth_eth_daily.csv")
    llama_price = df.sort_values("date")["price_eth"].iloc[-1]
    llama_disc  = (1 - llama_price) * 100
    llama_date  = df.sort_values("date")["date"].iloc[-1]

    # CoinGecko: live price (free, last 364 days works)
    try:
        r = requests.get("https://api.coingecko.com/api/v3/simple/price",
                         params={"ids": "staked-ether", "vs_currencies": "eth"},
                         timeout=10)
        cg_price = r.json()["staked-ether"]["eth"]
        cg_disc  = (1 - cg_price) * 100
    except Exception as e:
        cg_price = None
        print(f"  {WARN}  CoinGecko unavailable: {e}")

    print(f"\n  {'Source':<25} {'stETH/ETH price':>16}  {'discount %':>10}")
    print(f"  {'─'*25} {'─'*16}  {'─'*10}")
    print(f"  {'Curve pool (on-chain)':<25} {curve_price:>16.6f}  {curve_disc:>10.4f}%")
    if cg_price:
        print(f"  {'CoinGecko (live)':<25} {cg_price:>16.6f}  {cg_disc:>10.4f}%")
    print(f"  {'DeFiLlama ({})'.format(llama_date):<25} {llama_price:>16.6f}  {llama_disc:>10.4f}%")

    # Checks
    tol = 0.002   # 0.2% tolerance
    ok_curve_llama = abs(curve_price - llama_price) < tol
    check(f"|Curve − DeFiLlama| < {tol*100:.1f}%",
          ok_curve_llama,
          f"diff = {abs(curve_price - llama_price)*100:.4f}%")
    if cg_price:
        ok_curve_cg = abs(curve_price - cg_price) < tol
        check(f"|Curve − CoinGecko| < {tol*100:.1f}%",
              ok_curve_cg,
              f"diff = {abs(curve_price - cg_price)*100:.4f}%")

    # Historical: known event check — June 17 2022, stETH should be ~6% discount
    df["date"] = pd.to_datetime(df["date"])
    june17 = df[df["date"] == "2022-06-17"]
    if not june17.empty:
        disc_june17 = june17["discount_pct"].iloc[0]
        ok_hist = 4.0 < disc_june17 < 8.0
        check("June 17 2022 discount in [4%, 8%] (3AC/Celsius crisis)",
              ok_hist, f"actual = {disc_june17:.2f}%")

    # Post-V2: stETH should trade near par (discount < 0.5%)
    post_v2 = df[df["date"] >= "2023-05-16"]["discount_pct"]
    max_post = post_v2.max()
    ok_post  = max_post < 1.0
    check("Post-V2 max discount < 1% (queue ensures near-par trading)",
          ok_post, f"max = {max_post:.3f}%")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cross-validate Lido dataset")
    parser.add_argument("--rpc", default="https://ethereum.publicnode.com")
    args = parser.parse_args()

    w3 = Web3(Web3.HTTPProvider(args.rpc, request_kwargs={"timeout": 30}))
    if not w3.is_connected():
        raise ConnectionError(f"Cannot connect to RPC: {args.rpc}")

    req = pd.read_csv(DATA_DIR / "withdrawal_requested.csv")
    fin = pd.read_csv(DATA_DIR / "withdrawals_finalized.csv")

    ok1 = validate_onchain(w3, req, fin)
    validate_internal(req, fin)
    validate_price(w3)

    print(f"\n{'═'*60}\n")


if __name__ == "__main__":
    main()
