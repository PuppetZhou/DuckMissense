import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoModel



# main model of this work: ESM2 + BiLSTM + downstream classifier.
class MissenseESMBiLSTM(nn.Module):
    """
    ref/alt 双路结构：
    ESM2 residue embedding + AAIndex 数值特征 -> BiLSTM -> 突变位点向量 -> MLP。

    forward 输入 batch:
      ref_tokens / alt_tokens: tokenizer 输出的 dict(input_ids, attention_mask, ...)
      ref_aaindex / alt_aaindex: [B, L, K]，已 padding 的 AAIndex 特征
      lengths: [B]，每条窗口真实残基数
      mut_idx: [B]，突变位点在窗口内的 0-based 下标

    输出:
      logits: [B]，直接用于 BCEWithLogitsLoss
    """

    def __init__(
        self,
        plm_name="facebook/esm2_t33_650M_UR50D",
        aaindex_dim=6,
        lstm_hidden=512,
        proj_dim=256,
        dropout=0.2,
        freeze_esm=True,
        local_files_only=True,
        classifier=None,
    ):
        super().__init__()

        self.esm = AutoModel.from_pretrained(
            plm_name,
            local_files_only=local_files_only,
        )
        esm_dim = self.esm.config.hidden_size

        if freeze_esm:
            for param in self.esm.parameters():
                param.requires_grad = False

        self.encoder = nn.LSTM(
            input_size=esm_dim + aaindex_dim,
            hidden_size=lstm_hidden,
            batch_first=True,
            bidirectional=True,
        )

        encoded_dim = lstm_hidden * 2
        self.encoder_norm = nn.LayerNorm(encoded_dim)
        self.encoder_dropout = nn.Dropout(dropout)
        self.reduce = nn.Sequential(
            nn.Linear(encoded_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.classifier_input_dim = proj_dim * 4
        self.pair_norm = nn.LayerNorm(self.classifier_input_dim)
        self.classifier = classifier or MLPClassifier(
            input_dim=self.classifier_input_dim,
            hidden_dim=max(proj_dim // 2, 64),
            dropout=dropout,
        )

    def _strip_esm_special_tokens(self, tokens, lengths):
        # ESM2 序列前有 <cls>，残基 embedding 从位置 1 开始。
        out = self.esm(**tokens).last_hidden_state
        max_len = int(lengths.max().item())
        residue_emb = out.new_zeros(out.size(0), max_len, out.size(-1))

        for i, length in enumerate(lengths.tolist()):
            residue_emb[i, :length] = out[i, 1 : 1 + length]
        return residue_emb

    def _encode_one_branch(self, tokens, aaindex, lengths, mut_idx):
        esm_emb = self._strip_esm_special_tokens(tokens, lengths)
        fused = torch.cat([esm_emb, aaindex.to(esm_emb.device)], dim=-1)

        packed = pack_padded_sequence(
            fused,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        encoded, _ = self.encoder(packed)
        encoded, _ = pad_packed_sequence(
            encoded,
            batch_first=True,
            total_length=fused.size(1),
        )

        row = torch.arange(encoded.size(0), device=encoded.device)
        col = mut_idx.to(encoded.device).clamp(max=encoded.size(1) - 1)
        mut_vec = self.encoder_norm(encoded[row, col])
        mut_vec = self.encoder_dropout(mut_vec)
        return self.reduce(mut_vec)

    def forward(self, batch):
        ref_vec = self._encode_one_branch(
            batch["ref_tokens"],
            batch["ref_aaindex"],
            batch["lengths"],
            batch["mut_idx"],
        )
        alt_vec = self._encode_one_branch(
            batch["alt_tokens"],
            batch["alt_aaindex"],
            batch["lengths"],
            batch["mut_idx"],
        )

        pair_feature = torch.cat(
            [ref_vec, alt_vec, alt_vec - ref_vec, torch.abs(alt_vec - ref_vec)],
            dim=-1,
        )
        return self.classifier(self.pair_norm(pair_feature)).squeeze(-1)


# 下游分类器，尝试三种情况
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max(hidden_dim // 2, 32)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(hidden_dim // 2, 32), 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

class CNNPairClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.45, pair_channels=4):
        super().__init__()
        if input_dim % pair_channels != 0:
            raise ValueError("input_dim must be divisible by pair_channels")

        proj_dim = input_dim // pair_channels
        conv_dim = max(hidden_dim, 64)
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Unflatten(1, (pair_channels, proj_dim)),
            nn.Conv1d(pair_channels, conv_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(conv_dim, conv_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.LayerNorm(conv_dim),
            nn.Dropout(dropout),
            nn.Linear(conv_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class GatedResidualClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.45):
        super().__init__()
        gated_dim = max(hidden_dim, 64)
        self.norm = nn.LayerNorm(input_dim)
        self.value = nn.Sequential(
            nn.Linear(input_dim, gated_dim),
            nn.GELU(),
        )
        self.gate = nn.Sequential(
            nn.Linear(input_dim, gated_dim),
            nn.Sigmoid(),
        )
        self.residual = nn.Linear(input_dim, gated_dim)
        self.out = nn.Sequential(
            nn.LayerNorm(gated_dim),
            nn.Dropout(dropout),
            nn.Linear(gated_dim, max(gated_dim // 2, 32)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(gated_dim // 2, 32), 1),
        )

    def forward(self, x):
        x = self.norm(x)
        hidden = self.value(x) * self.gate(x) + self.residual(x)
        return self.out(hidden).squeeze(-1)

