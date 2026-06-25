#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from aaindex import AAINDEX_DIM, encode_aaindex
from model import (
    CNNPairClassifier,
    GatedResidualClassifier,
    MLPClassifier,
    MissenseESMBiLSTM,
)


ROOT = Path(__file__).resolve().parents[1]
AA_SET = set("ACDEFGHIKLMNPQRSTVWY")
VARIANT_RE = re.compile(r"^([A-Z*])(\d+)([A-Z*])$")
DEFAULT_PLM_NAME = "facebook/esm2_t33_650M_UR50D"
DEFAULT_UPSTREAM = 255
DEFAULT_DOWNSTREAM = 256


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Predict missense pathogenicity from a reference protein sequence, "
            "1-based mutation position, and mutated amino acid."
        )
    )
    parser.add_argument("--model", required=True, help="Path to model checkpoint .pt file.")
    parser.add_argument(
        "--input-csv",
        default=None,
        help=(
            "Optional CSV for batch prediction. Required columns can be "
            "ref_seq/sequence plus either protein_variant/variant or "
            "position/pos/mut_pos and alt_aa/alt/mut_aa."
        ),
    )
    parser.add_argument("--ref-seq", default=None, help="Reference protein sequence for one prediction.")
    parser.add_argument("--position", type=int, default=None, help="1-based mutation position.")
    parser.add_argument("--alt-aa", default=None, help="Mutated amino acid, for example Q.")
    parser.add_argument("--sample-id", default=None, help="Optional sample ID for one prediction.")
    parser.add_argument("--output", default=None, help="Optional output CSV path. If omitted, print to stdout.")
    parser.add_argument("--threshold", type=float, default=None, help="Decision threshold. Defaults to checkpoint config or 0.5.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--upstream", type=int, default=DEFAULT_UPSTREAM)
    parser.add_argument("--downstream", type=int, default=DEFAULT_DOWNSTREAM)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow transformers to download the ESM2 model/tokenizer if not available locally.",
    )
    return parser.parse_args()


def clean_seq(seq):
    return "".join(aa if aa in AA_SET else "X" for aa in str(seq).upper())


def make_window_from_position(ref_seq, position, alt_aa, upstream, downstream):
    if position is None or int(position) <= 0:
        raise ValueError(f"position must be a positive 1-based integer, got {position!r}")

    alt_aa = str(alt_aa).strip().upper()
    if len(alt_aa) != 1 or alt_aa not in AA_SET:
        raise ValueError(f"alt_aa must be one of {''.join(sorted(AA_SET))}, got {alt_aa!r}")

    ref_seq = clean_seq(ref_seq)
    pos0 = int(position) - 1
    if pos0 >= len(ref_seq):
        raise ValueError(
            f"position {position} is outside the reference sequence length {len(ref_seq)}"
        )

    ref_aa = ref_seq[pos0]
    alt_seq = list(ref_seq)
    alt_seq[pos0] = alt_aa
    alt_seq = "".join(alt_seq)

    start = max(0, pos0 - upstream)
    end = min(len(ref_seq), pos0 + downstream + 1)
    return {
        "ref_seq": ref_seq,
        "position": int(position),
        "ref_aa": ref_aa,
        "alt_aa": alt_aa,
        "protein_variant": f"{ref_aa}{int(position)}{alt_aa}",
        "ref_window": ref_seq[start:end],
        "alt_window": alt_seq[start:end],
        "mut_idx": pos0 - start,
    }


def parse_protein_variant(variant):
    match = VARIANT_RE.match(str(variant).strip().upper())
    if match is None:
        raise ValueError(f"bad protein_variant: {variant!r}; expected format like R403Q")
    ref_aa, position, alt_aa = match.groups()
    if ref_aa == "*" or alt_aa == "*":
        raise ValueError(f"stop-gain/loss variants are not supported: {variant!r}")
    return ref_aa, int(position), alt_aa


def find_column(columns, candidates, required_name):
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(
        f"input CSV is missing {required_name}. Accepted columns: {', '.join(candidates)}"
    )


def read_prediction_inputs(args):
    if args.input_csv:
        df = pd.read_csv(args.input_csv)
        seq_col = find_column(df.columns, ["ref_seq", "sequence"], "reference sequence")
        pos_col = next((col for col in ["position", "pos", "mut_pos", "mutation_pos"] if col in df.columns), None)
        alt_col = next((col for col in ["alt_aa", "alt", "mut_aa", "mutated_aa"] if col in df.columns), None)
        variant_col = next((col for col in ["protein_variant", "variant"] if col in df.columns), None)
        id_col = next((col for col in ["sample_id", "id", "variant_id", "database_id"] if col in df.columns), None)

        if not ((pos_col is not None and alt_col is not None) or variant_col is not None):
            raise ValueError(
                "input CSV needs either position + alt_aa columns or a protein_variant column"
            )

        records = []
        errors = []
        for row_idx, row in df.iterrows():
            try:
                expected_ref_aa = None
                if pos_col is not None and alt_col is not None:
                    position = row[pos_col]
                    alt_aa = row[alt_col]
                    input_variant = None
                else:
                    expected_ref_aa, position, alt_aa = parse_protein_variant(row[variant_col])
                    input_variant = str(row[variant_col]).strip().upper()

                record = make_window_from_position(
                    row[seq_col],
                    position,
                    alt_aa,
                    args.upstream,
                    args.downstream,
                )
                if expected_ref_aa and record["ref_aa"] != expected_ref_aa:
                    raise ValueError(
                        f"variant reference AA mismatch: variant says {expected_ref_aa}, "
                        f"sequence has {record['ref_aa']} at position {position}"
                    )
                if input_variant:
                    record["protein_variant"] = input_variant
                record["sample_id"] = row[id_col] if id_col else row_idx
                records.append(record)
            except Exception as exc:
                errors.append(f"row {row_idx}: {exc}")

        if errors:
            preview = "\n".join(errors[:5])
            raise ValueError(f"failed to parse {len(errors)} input row(s):\n{preview}")
        return pd.DataFrame(records)

    if args.ref_seq is None or args.position is None or args.alt_aa is None:
        raise ValueError(
            "provide either --input-csv or all of --ref-seq, --position, and --alt-aa"
        )

    record = make_window_from_position(
        args.ref_seq,
        args.position,
        args.alt_aa,
        args.upstream,
        args.downstream,
    )
    record["sample_id"] = args.sample_id or "sample_0"
    return pd.DataFrame([record])


def build_classifier(input_dim, classifier_type, proj_dim, dropout):
    hidden_dim = max(proj_dim // 2, 64)
    if classifier_type == "mlp":
        return MLPClassifier(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
    if classifier_type == "cnn":
        return CNNPairClassifier(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
    if classifier_type == "gated":
        return GatedResidualClassifier(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
    raise ValueError(f"Unsupported classifier_type in checkpoint: {classifier_type}")


def load_model(checkpoint_path, device, local_files_only):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint.get("config", {})

    classifier_type = cfg.get("classifier_type", "mlp")
    proj_dim = int(cfg.get("proj_dim", 64))
    dropout = float(cfg.get("dropout", 0.5))
    lstm_hidden = int(cfg.get("lstm_hidden", 256))
    plm_name = cfg.get("plm_name", DEFAULT_PLM_NAME)
    freeze_esm = bool(cfg.get("freeze_esm", True))

    classifier = build_classifier(proj_dim * 4, classifier_type, proj_dim, dropout)
    model = MissenseESMBiLSTM(
        plm_name=plm_name,
        aaindex_dim=AAINDEX_DIM,
        lstm_hidden=lstm_hidden,
        proj_dim=proj_dim,
        dropout=dropout,
        freeze_esm=freeze_esm,
        local_files_only=local_files_only,
        classifier=classifier,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, cfg


def pad_aaindex(tensors, max_len):
    dim = tensors[0].shape[-1]
    out = torch.zeros(len(tensors), max_len, dim, dtype=torch.float32)
    for idx, tensor in enumerate(tensors):
        out[idx, : tensor.shape[0]] = tensor
    return out


def make_model_batch(df, tokenizer, device):
    lengths = torch.tensor(df["ref_window"].str.len().to_numpy(), dtype=torch.long)
    max_len = int(lengths.max())

    ref_aaindex = [
        torch.tensor(encode_aaindex(seq), dtype=torch.float32)
        for seq in df["ref_window"].tolist()
    ]
    alt_aaindex = [
        torch.tensor(encode_aaindex(seq), dtype=torch.float32)
        for seq in df["alt_window"].tolist()
    ]

    batch = {
        "ref_tokens": tokenizer(df["ref_window"].tolist(), padding=True, return_tensors="pt"),
        "alt_tokens": tokenizer(df["alt_window"].tolist(), padding=True, return_tensors="pt"),
        "ref_aaindex": pad_aaindex(ref_aaindex, max_len),
        "alt_aaindex": pad_aaindex(alt_aaindex, max_len),
        "lengths": lengths,
        "mut_idx": torch.tensor(df["mut_idx"].to_numpy(), dtype=torch.long),
    }
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


@torch.no_grad()
def predict(model, df, tokenizer, device, batch_size):
    logits = []
    for start in range(0, len(df), batch_size):
        part = df.iloc[start : start + batch_size]
        batch = make_model_batch(part, tokenizer, device)
        batch_logits = model(batch)
        logits.extend(batch_logits.detach().cpu().numpy().tolist())
    logits = np.array(logits, dtype=np.float32)
    probs = 1.0 / (1.0 + np.exp(-logits))
    return logits, probs


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else
        args.device
    )
    local_files_only = not args.allow_download

    input_df = read_prediction_inputs(args)
    model, cfg = load_model(args.model, device, local_files_only=local_files_only)
    threshold = args.threshold
    if threshold is None:
        threshold = float(cfg.get("threshold", 0.5))

    plm_name = cfg.get("plm_name", DEFAULT_PLM_NAME)
    tokenizer = AutoTokenizer.from_pretrained(
        plm_name,
        local_files_only=local_files_only,
    )

    logits, probs = predict(model, input_df, tokenizer, device, args.batch_size)
    out_df = input_df[
        [
            "sample_id",
            "position",
            "ref_aa",
            "alt_aa",
            "protein_variant",
            "mut_idx",
        ]
    ].copy()
    out_df["window_length"] = input_df["ref_window"].str.len().to_numpy()
    out_df["logit"] = logits
    out_df["probability"] = probs
    out_df["threshold"] = threshold
    out_df["pred_label"] = (probs >= threshold).astype(int)
    out_df["prediction"] = np.where(out_df["pred_label"] == 1, "pathogenic", "benign")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(output_path, index=False)
        print(f"Saved predictions to: {output_path}")
    else:
        print(out_df.to_csv(index=False), end="")


if __name__ == "__main__":
    main()
