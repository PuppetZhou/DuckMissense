import re

import numpy as np
import torch


AA_SET = set("ACDEFGHIKLMNPQRSTVWY")
VARIANT_RE = re.compile(r"^([A-Z*])(\d+)([A-Z*])$")


def clean_seq(seq):
    """Replace non-standard amino acids with X."""
    return "".join(aa if aa in AA_SET else "X" for aa in str(seq).upper())


def parse_protein_variant(variant):
    """Parse a protein variant string like R403Q into (ref_aa, position, alt_aa)."""
    match = VARIANT_RE.match(str(variant).strip().upper())
    if match is None:
        raise ValueError(f"bad protein_variant: {variant!r}; expected format like R403Q")
    ref_aa, position, alt_aa = match.groups()
    if ref_aa == "*" or alt_aa == "*":
        raise ValueError(f"stop-gain/loss variants are not supported: {variant!r}")
    return ref_aa, int(position), alt_aa


def make_window_from_position(ref_seq, position, alt_aa, upstream, downstream, allow_x_alt=False):
    """Extract ref/alt windows around a 1-based mutation position."""
    if position is None or int(position) <= 0:
        raise ValueError(f"position must be a positive 1-based integer, got {position!r}")

    alt_aa = str(alt_aa).strip().upper()
    valid_alts = AA_SET if not allow_x_alt else AA_SET | {"X"}
    if len(alt_aa) != 1 or alt_aa not in valid_alts:
        raise ValueError(f"alt_aa must be one of {''.join(sorted(valid_alts))}, got {alt_aa!r}")

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


def make_window_from_variant(ref_seq, variant, upstream, downstream, strict_alt=True):
    """Parse a variant string and return (ref_window, alt_window, mut_idx)."""
    match = VARIANT_RE.match(str(variant).strip().upper())
    if match is None:
        raise ValueError(f"bad variant: {variant}")
    ref_aa, position, alt_aa = match.groups()
    if strict_alt:
        ref_aa, position, alt_aa = parse_protein_variant(variant)
    else:
        if ref_aa not in AA_SET:
            ref_aa = "X"
        if alt_aa not in AA_SET:
            alt_aa = "X"
        position = int(position)

    window = make_window_from_position(
        ref_seq, position, alt_aa, upstream, downstream, allow_x_alt=not strict_alt
    )
    return window["ref_window"], window["alt_window"], window["mut_idx"]


def pad_aaindex(tensors, max_len):
    """Pad a list of [L, K] AAIndex tensors to [B, max_len, K]."""
    dim = tensors[0].shape[-1]
    out = torch.zeros(len(tensors), max_len, dim, dtype=torch.float32)
    for idx, tensor in enumerate(tensors):
        out[idx, : tensor.shape[0]] = tensor
    return out
