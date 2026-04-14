#!/usr/bin/env python3
"""Convert MovieLens-100K to SELFRec's yelp2018-style implicit format.

Output files:
- dataset/movielens/train.txt
- dataset/movielens/test.txt

Line format (space-separated), matching `dataset/yelp2018/*.txt`:
    <user_id> <item_id> 1

Split strategy:
- leave-one-out by timestamp per user (latest interaction -> test)
- remaining interactions -> train
"""

from __future__ import annotations

import argparse
import csv
import io
import zipfile
from collections import defaultdict
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

ML100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"


def load_from_zip_bytes(payload: bytes) -> list[tuple[int, int, int, int]]:
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        with zf.open("ml-100k/u.data") as fp:
            rows: list[tuple[int, int, int, int]] = []
            reader = csv.reader(io.TextIOWrapper(fp, encoding="utf-8"), delimiter="\t")
            for row in reader:
                if len(row) != 4:
                    continue
                user, item, rating, ts = row
                rows.append((int(user), int(item), int(rating), int(ts)))
    return rows


def load_from_u_data(path: Path) -> list[tuple[int, int, int, int]]:
    rows: list[tuple[int, int, int, int]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) != 4:
                continue
            user, item, rating, ts = row
            rows.append((int(user), int(item), int(rating), int(ts)))
    return rows


def load_movielens100k(zip_path: Path | None, u_data_path: Path | None) -> list[tuple[int, int, int, int]]:
    if u_data_path is not None:
        if not u_data_path.exists():
            raise FileNotFoundError(f"u.data not found: {u_data_path}")
        return load_from_u_data(u_data_path)

    if zip_path is not None:
        if not zip_path.exists():
            raise FileNotFoundError(f"zip file not found: {zip_path}")
        return load_from_zip_bytes(zip_path.read_bytes())

    try:
        with urlopen(ML100K_URL) as response:
            payload = response.read()
        return load_from_zip_bytes(payload)
    except URLError as e:
        raise RuntimeError(
            "Failed to download ml-100k automatically. "
            "Please pass --zip-path /path/to/ml-100k.zip "
            "or --u-data-path /path/to/u.data"
        ) from e


def remap_ids(interactions: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    user_ids = sorted({u for u, _, _, _ in interactions})
    item_ids = sorted({i for _, i, _, _ in interactions})
    user_map = {u: idx for idx, u in enumerate(user_ids)}
    item_map = {i: idx for idx, i in enumerate(item_ids)}
    return [(user_map[u], item_map[i], r, ts) for u, i, r, ts in interactions]


def leave_one_out_split(
    interactions: list[tuple[int, int, int, int]],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    by_user: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for u, i, _r, ts in interactions:
        by_user[u].append((i, ts, len(by_user[u])))

    train: list[tuple[int, int]] = []
    test: list[tuple[int, int]] = []

    for u, records in by_user.items():
        records = sorted(records, key=lambda x: (x[1], x[2]))
        if len(records) == 1:
            train.append((u, records[0][0]))
            continue
        for i, _ts, _idx in records[:-1]:
            train.append((u, i))
        test.append((u, records[-1][0]))

    train.sort(key=lambda x: (x[0], x[1]))
    test.sort(key=lambda x: (x[0], x[1]))
    return train, test


def write_split(path: Path, rows: list[tuple[int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for u, i in rows:
            f.write(f"{u} {i} 1\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset") / "movielens",
        help="Output directory for train.txt/test.txt",
    )
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=None,
        help="Local path to ml-100k.zip (optional)",
    )
    parser.add_argument(
        "--u-data-path",
        type=Path,
        default=None,
        help="Local path to u.data (optional)",
    )
    args = parser.parse_args()

    raw = load_movielens100k(args.zip_path, args.u_data_path)
    mapped = remap_ids(raw)
    train, test = leave_one_out_split(mapped)

    write_split(args.output_dir / "train.txt", train)
    write_split(args.output_dir / "test.txt", test)

    print(f"Saved train: {args.output_dir / 'train.txt'} ({len(train)} rows)")
    print(f"Saved test : {args.output_dir / 'test.txt'} ({len(test)} rows)")


if __name__ == "__main__":
    main()
