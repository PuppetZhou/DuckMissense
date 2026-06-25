import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from aaindex import encode_aaindex


class MissenseDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            "ref_seq": row.ref_window,
            "alt_seq": row.alt_window,
            "ref_aaindex": torch.tensor(encode_aaindex(row.ref_window)),
            "alt_aaindex": torch.tensor(encode_aaindex(row.alt_window)),
            "length": len(row.ref_window),
            "mut_idx": int(row.mut_idx),
            "label": float(row.ClinicalSig),
        }


class MissenseCollator:
    def __init__(self, plm_name="facebook/esm2_t33_650M_UR50D", local_files_only=True):
        self.tokenizer = AutoTokenizer.from_pretrained(plm_name, local_files_only=local_files_only)

    def _pad_aaindex(self, tensors, max_len):
        dim = tensors[0].shape[-1]
        out = torch.zeros(len(tensors), max_len, dim, dtype=torch.float32)
        for i, tensor in enumerate(tensors):
            out[i, : tensor.shape[0]] = tensor
        return out

    def __call__(self, batch):
        lengths = torch.tensor([x["length"] for x in batch], dtype=torch.long)
        max_len = int(lengths.max())

        return {
            "ref_tokens": self.tokenizer([x["ref_seq"] for x in batch], padding=True, return_tensors="pt"),
            "alt_tokens": self.tokenizer([x["alt_seq"] for x in batch], padding=True, return_tensors="pt"),
            "ref_aaindex": self._pad_aaindex([x["ref_aaindex"] for x in batch], max_len),
            "alt_aaindex": self._pad_aaindex([x["alt_aaindex"] for x in batch], max_len),
            "lengths": lengths,
            "mut_idx": torch.tensor([x["mut_idx"] for x in batch], dtype=torch.long),
            "labels": torch.tensor([x["label"] for x in batch], dtype=torch.float32),
        }
