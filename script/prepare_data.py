import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from data_utils import make_window_from_variant


ROOT = Path(__file__).resolve().parents[1]

INPUT_CSV = ROOT / "data/proceed/variant_clean_final.csv"
OUT_DIR = ROOT / "data/proceed/splits"
SEED = 42
UPSTREAM = 255
DOWNSTREAM = 256
POS_TO_NEG_RATIO = 1 # 最终正负样本比例
TRAIN_RATIO = 0.8 # 训练集比例
VAL_RATIO = 0.1 # 验证集比例
TEST_RATIO = 0.1 # 测试集比例


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(INPUT_CSV))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--upstream", type=int, default=UPSTREAM)
    parser.add_argument("--downstream", type=int, default=DOWNSTREAM)
    parser.add_argument("--neg-per-pos", type=int, default=POS_TO_NEG_RATIO)
    parser.add_argument("--train-ratio", type=float, default=TRAIN_RATIO)
    parser.add_argument("--val-ratio", type=float, default=VAL_RATIO)
    parser.add_argument("--test-ratio", type=float, default=TEST_RATIO)
    parser.add_argument(
        "--total-count",
        type=int,
        default=None,
        help="Total number of samples after sampling and before split. "
             "If set, the dataset is downsampled to this size while keeping "
             "the pos:neg ratio. Splits remain 8:1:1 stratified.",
    )
    parser.add_argument(
        "--max-per-gene",
        type=int,
        default=None,
        help="Optional cap on rows kept per gene_symbol before label sampling. "
             "Use this to reduce dominance from genes with many variants. "
             "If omitted, no gene cap is applied.",
    )
    return parser.parse_args()


def add_windows(df, upstream, downstream):
    rows = [
        make_window_from_variant(row.ref_seq, row.protein_variant, upstream, downstream, strict_alt=False)
        for row in df.itertuples(index=False)
    ]
    df = df.copy()
    df["ref_window"] = [x[0] for x in rows]
    df["alt_window"] = [x[1] for x in rows]
    df["mut_idx"] = [x[2] for x in rows]
    return df.drop(columns=["ref_seq"])


def limit_per_gene(df, max_per_gene, seed):
    if max_per_gene is None:
        return df
    if max_per_gene <= 0:
        raise ValueError("--max-per-gene must be positive")

    df = df.copy()
    gene_key = df["gene_symbol"].fillna("__MISSING_GENE__")
    df["_gene_key_for_cap"] = gene_key
    capped = (
        df.groupby("_gene_key_for_cap", group_keys=False)
        .apply(lambda part: part.sample(n=min(len(part), max_per_gene), random_state=seed))
        .drop(columns=["_gene_key_for_cap"], errors="ignore")
    )
    return capped.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def sample_labels(df, neg_per_pos, seed, total_count=None):
    pos = df[df["ClinicalSig"] == 1]
    neg = df[df["ClinicalSig"] == 0]

    if total_count is not None:
        # Keep pos:neg ratio while targeting total_count.
        n_pos = total_count // (neg_per_pos + 1)
        n_neg = total_count - n_pos
    else:
        n_pos = min(len(pos), len(neg) // neg_per_pos)
        n_neg = n_pos * neg_per_pos

    n_pos = min(n_pos, len(pos))
    n_neg = min(n_neg, len(neg))

    df = pd.concat(
        [
            pos.sample(n=n_pos, random_state=seed),
            neg.sample(n=n_neg, random_state=seed),
        ],
        ignore_index=True,
    )
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    usecols = ["gene_symbol", "ClinicalSig", "protein_variant", "source", "database_id", "mapping_id", "ref_seq"]
    df = pd.read_csv(args.input, usecols=usecols)
    df = limit_per_gene(df, args.max_per_gene, args.seed)
    df = add_windows(df, args.upstream, args.downstream)
    df = sample_labels(df, args.neg_per_pos, args.seed, total_count=args.total_count)

    holdout_ratio = args.val_ratio + args.test_ratio
    test_ratio_in_holdout = args.test_ratio / holdout_ratio
    train, tmp = train_test_split(df, test_size=holdout_ratio, stratify=df["ClinicalSig"], random_state=args.seed)
    val, test = train_test_split(tmp, test_size=test_ratio_in_holdout, stratify=tmp["ClinicalSig"], random_state=args.seed)

    train.to_csv(out_dir / "train.csv", index=False)
    val.to_csv(out_dir / "val.csv", index=False)
    test.to_csv(out_dir / "test.csv", index=False)

    for name, part in [("train", train), ("val", val), ("test", test)]:
        top_gene = part["gene_symbol"].value_counts(dropna=False).head(1)
        max_gene = int(top_gene.iloc[0]) if len(top_gene) else 0
        max_gene_name = str(top_gene.index[0]) if len(top_gene) else "NA"
        print(
            f"{name}: n={len(part)} pos={int(part.ClinicalSig.sum())} "
            f"neg={int((part.ClinicalSig == 0).sum())} "
            f"max_gene={max_gene_name}:{max_gene}"
        )


if __name__ == "__main__":
    main()
