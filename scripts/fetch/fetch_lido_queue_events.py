"""
Fetch Lido WithdrawalQueueERC721 events from Ethereum mainnet via web3.py.
"""

import argparse
import csv
import time
from pathlib import Path

import requests
from web3 import Web3

from environ.constants import DATA_PATH as DATA_DIR

# Contract
WQ_ADDR = "0x889edC2eDab5f40e902b864aD4d7AdE8E412F9B1"

# Event topics  (keccak256 of canonical signatures)
TOPIC_REQUESTED = "0xf0cb471f23fb74ea44b8252eb1881a2dca546288d9f6e90d1a0e82fe0ed342ab"
TOPIC_FINALIZED = "0x197874c72af6a06fb0aa4fab45fd39c7cb61ac0992159872dc3295207da7e9eb"

# Minimal ABI – only what we need
WQ_ABI = [
    {
        "name": "WithdrawalRequested",
        "type": "event",
        "inputs": [
            {"name": "requestId", "type": "uint256", "indexed": True},
            {"name": "requestor", "type": "address", "indexed": True},
            {"name": "owner", "type": "address", "indexed": True},
            {"name": "amountOfStETH", "type": "uint256", "indexed": False},
            {"name": "amountOfShares", "type": "uint256", "indexed": False},
        ],
    },
    {
        "name": "WithdrawalsFinalized",
        "type": "event",
        "inputs": [
            {"name": "from", "type": "uint256", "indexed": True},
            {"name": "to", "type": "uint256", "indexed": True},
            {"name": "amountOfETHLocked", "type": "uint256", "indexed": False},
            {"name": "sharesToBurn", "type": "uint256", "indexed": False},
            {"name": "maxShareRate", "type": "uint256", "indexed": False},
        ],
    },
]

# Contract deployed in the Lido V2 upgrade (May 15, 2023)
# First WithdrawalRequested event is at block 17,266,004 (request IDs 1–69)
DEFAULT_START_BLOCK = 17_172_547  # true deployment block; first event at 17,266,004


# Helpers


def connect(rpc_url: str) -> Web3:
    w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": 30}))
    if not w3.is_connected():
        raise ConnectionError(f"Cannot connect to RPC: {rpc_url}")
    print(f"Connected to RPC – latest block: {w3.eth.block_number:,}")
    return w3


def get_block_timestamp(w3: Web3, block_number: int) -> int:
    return w3.eth.get_block(block_number)["timestamp"]


def fetch_logs_chunked(
    w3: Web3,
    address: str,
    topic: str,
    from_block: int,
    to_block: int,
    chunk_size: int = 2_000,
    retry_delay: float = 2.0,
    max_retries: int = 3,
):
    """Yield raw log dicts for a single topic, chunked to avoid RPC limits."""
    current = from_block
    while current <= to_block:
        end = min(current + chunk_size - 1, to_block)
        for attempt in range(max_retries):
            try:
                logs = w3.eth.get_logs(
                    {
                        "address": address,
                        "fromBlock": current,
                        "toBlock": end,
                        "topics": [topic],
                    }
                )
                yield from logs
                break
            except Exception as exc:
                if attempt < max_retries - 1:
                    print(
                        f"  Retry {attempt+1}/{max_retries} for blocks {current}-{end}: {exc}"
                    )
                    time.sleep(retry_delay * (attempt + 1))
                else:
                    print(
                        f"  Skipping blocks {current}-{end} after {max_retries} failures: {exc}"
                    )
        current = end + 1


# Decoders


def _log_data_bytes(log) -> bytes:
    """Extract raw data bytes from a log entry. HexBytes converts cleanly via bytes()."""
    return bytes(log["data"])


def decode_requested(log, w3: Web3):
    """
    Parse a WithdrawalRequested log.
    Indexed topics: requestId (uint256), requestor (address), owner (address)
    Data (non-indexed): amountOfStETH (uint256), amountOfShares (uint256)
    """
    request_id = int(log["topics"][1].hex(), 16)
    requestor = w3.to_checksum_address("0x" + log["topics"][2].hex()[-40:])
    owner = w3.to_checksum_address("0x" + log["topics"][3].hex()[-40:])
    data = _log_data_bytes(log)
    amount_steth = int.from_bytes(data[0:32], "big")
    amount_shares = int.from_bytes(data[32:64], "big")
    return {
        "request_id": request_id,
        "requestor": requestor,
        "owner": owner,
        "amount_steth": amount_steth / 1e18,
        "amount_shares": amount_shares / 1e18,
        "block_number": log["blockNumber"],
        "tx_hash": log["transactionHash"].hex(),
    }


def decode_finalized(log):
    """
    Parse a WithdrawalsFinalized log.
    Indexed topics: from (uint256), to (uint256)
    Data (non-indexed): amountOfETHLocked (uint256), sharesToBurn (uint256),
                        maxShareRate (uint256, ray = 1e27)
    """
    from_id = int(log["topics"][1].hex(), 16)
    to_id = int(log["topics"][2].hex(), 16)
    data = _log_data_bytes(log)
    amount_eth_locked = int.from_bytes(data[0:32], "big")
    shares_to_burn = int.from_bytes(data[32:64], "big")
    max_share_rate = int.from_bytes(data[64:96], "big")
    return {
        "from_request_id": from_id,
        "to_request_id": to_id,
        "requests_finalized": to_id - from_id + 1,
        "amount_eth_locked": amount_eth_locked / 1e18,
        "shares_to_burn": shares_to_burn / 1e18,
        "max_share_rate": max_share_rate / 1e27,  # ray (27 decimals)
        "block_number": log["blockNumber"],
        "tx_hash": log["transactionHash"].hex(),
    }


# Batch block-timestamp fetching

_block_ts_cache: dict[int, int] = {}
_BATCH_SIZE = 500


def _batch_get_timestamps(rpc_url: str, block_numbers: list[int]) -> dict[int, int]:
    """
    Fetch block timestamps for a list of block numbers using JSON-RPC batch.
    Returns {block_number: unix_timestamp}.
    """
    result: dict[int, int] = {}
    for i in range(0, len(block_numbers), _BATCH_SIZE):
        chunk = block_numbers[i : i + _BATCH_SIZE]
        payload = [
            {
                "jsonrpc": "2.0",
                "method": "eth_getBlockByNumber",
                "params": [hex(bn), False],  # False = don't fetch full tx list
                "id": j,
            }
            for j, bn in enumerate(chunk)
        ]
        for attempt in range(3):
            try:
                resp = requests.post(rpc_url, json=payload, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                if isinstance(data, dict):  # error response
                    raise RuntimeError(f"RPC batch error: {data}")
                for item, bn in zip(data, chunk):
                    blk = item.get("result") or {}
                    if blk and blk.get("timestamp"):
                        result[bn] = int(blk["timestamp"], 16)
                break
            except Exception as exc:
                if attempt < 2:
                    print(f"  Batch attempt {attempt+1}/3 failed: {exc}")
                    time.sleep(2**attempt)
                else:
                    print(f"  Batch permanently failed for {len(chunk)} blocks: {exc}")
    return result


def resolve_timestamps(w3: Web3, rows: list[dict], rpc_url: str = "") -> list[dict]:
    """Add 'timestamp' field to each row using JSON-RPC batch requests."""
    unique_blocks = sorted({r["block_number"] for r in rows} - _block_ts_cache.keys())
    if unique_blocks:
        print(
            f"  Batch-fetching timestamps for {len(unique_blocks):,} blocks "
            f"({len(unique_blocks) // _BATCH_SIZE + 1} batch(es)) …"
        )
        url = rpc_url or w3.provider.endpoint_uri
        fetched = _batch_get_timestamps(url, unique_blocks)
        _block_ts_cache.update(fetched)
        failed = sorted(set(unique_blocks) - fetched.keys())
        if failed:
            failed_path = DATA_DIR / "failed_timestamp_blocks.txt"
            with open(failed_path, "a") as f:
                f.write("\n".join(str(b) for b in failed) + "\n")
            print(f"  WARNING: {len(failed)} blocks missing timestamps → {failed_path}")
    for r in rows:
        r["timestamp"] = _block_ts_cache.get(r["block_number"], 0)
    return rows


# Writers

REQUESTED_FIELDS = [
    "request_id",
    "requestor",
    "owner",
    "amount_steth",
    "amount_shares",
    "block_number",
    "timestamp",
    "tx_hash",
]
FINALIZED_FIELDS = [
    "from_request_id",
    "to_request_id",
    "requests_finalized",
    "amount_eth_locked",
    "shares_to_burn",
    "max_share_rate",
    "block_number",
    "timestamp",
    "tx_hash",
]


def append_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def last_saved_block(path: Path, block_col: str = "block_number") -> int:
    """Return the highest block_number already saved, so we can resume."""
    if not path.exists():
        return 0
    max_block = 0
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                max_block = max(max_block, int(row[block_col]))
            except (ValueError, KeyError):
                pass
    return max_block


# Main


def main():
    parser = argparse.ArgumentParser(description="Fetch Lido withdrawal queue events")
    parser.add_argument(
        "--rpc",
        default="https://ethereum.publicnode.com",
        help="Ethereum JSON-RPC endpoint",
    )
    parser.add_argument(
        "--start-block",
        type=int,
        default=DEFAULT_START_BLOCK,
        help="First block to scan (default: %(default)s)",
    )
    parser.add_argument(
        "--end-block",
        type=int,
        default=None,
        help="Last block to scan (default: latest)",
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=10_000,
        help="Block range per getLogs call (default: %(default)s, max 50000 for publicnode)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=20_000,
        help="Rows to buffer before flushing to CSV (default: %(default)s)",
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    req_path = DATA_DIR / "withdrawal_requested.csv"
    fin_path = DATA_DIR / "withdrawals_finalized.csv"

    w3 = connect(args.rpc)
    end_block = args.end_block or w3.eth.block_number

    # Resume from last saved block
    req_start = max(args.start_block, last_saved_block(req_path) + 1)
    fin_start = max(args.start_block, last_saved_block(fin_path) + 1)

    print(f"\n=== WithdrawalRequested  ({req_start:,} → {end_block:,}) ===")
    buf_req: list[dict] = []
    total_req = 0
    for log in fetch_logs_chunked(
        w3, WQ_ADDR, TOPIC_REQUESTED, req_start, end_block, args.chunk
    ):
        buf_req.append(decode_requested(log, w3))
        if len(buf_req) >= args.batch:
            buf_req = resolve_timestamps(w3, buf_req, args.rpc)
            append_csv(req_path, buf_req, REQUESTED_FIELDS)
            total_req += len(buf_req)
            print(f"  Saved {total_req:,} requests so far …")
            buf_req = []
    if buf_req:
        buf_req = resolve_timestamps(w3, buf_req, args.rpc)
        append_csv(req_path, buf_req, REQUESTED_FIELDS)
        total_req += len(buf_req)
    print(f"  Total WithdrawalRequested saved: {total_req:,}")

    print(f"\n=== WithdrawalsFinalized ({fin_start:,} → {end_block:,}) ===")
    buf_fin: list[dict] = []
    total_fin = 0
    for log in fetch_logs_chunked(
        w3, WQ_ADDR, TOPIC_FINALIZED, fin_start, end_block, args.chunk
    ):
        buf_fin.append(decode_finalized(log))
        if len(buf_fin) >= args.batch:
            buf_fin = resolve_timestamps(w3, buf_fin, args.rpc)
            append_csv(fin_path, buf_fin, FINALIZED_FIELDS)
            total_fin += len(buf_fin)
            print(f"  Saved {total_fin:,} finalization events so far …")
            buf_fin = []
    if buf_fin:
        buf_fin = resolve_timestamps(w3, buf_fin, args.rpc)
        append_csv(fin_path, buf_fin, FINALIZED_FIELDS)
        total_fin += len(buf_fin)
    print(f"  Total WithdrawalsFinalized saved: {total_fin:,}")

    print("\nDone. Output files:")
    print(f"  {req_path}")
    print(f"  {fin_path}")


if __name__ == "__main__":
    main()
