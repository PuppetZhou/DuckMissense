# Downstream Classifier Comparison

## Experiment Setup

- Backbone: frozen `facebook/esm2_t33_650M_UR50D` + AAIndex + BiLSTM.
- Shared architecture hyperparameters: `LSTM_HIDDEN=256`, `PROJ_DIM=128`, `DROPOUT=0.45`.
- Shared training hyperparameters: `LR=5e-5`, `BATCH_SIZE=2`, `EPOCHS=3`, `FREEZE_ESM=True`, `USE_AMP=True`, `SEED=42`.
- Tuning split: `data/proceed/splits_tune_3000`, with train `n=2400`, val `n=300`, test `n=300`, all balanced 1:1 positive/negative.
- Model selection metric in `train.py`: best validation AUPRC.

## Classifiers

| Classifier | Class | Location | Description |
| --- | --- | --- | --- |
| MLP | `MLPClassifier` | `script/model.py` | LayerNorm + two-layer GELU MLP with dropout. |
| CNN | `CNNPairClassifier` | `script/model.py` | Reshapes `[ref, alt, diff, absdiff]` into 4 channels and applies 1D CNN over projected dimensions. |
| Gated | `GatedResidualClassifier` | `script/model.py` | Gated dense interaction with residual projection and dropout. |

## Validation Results

| Classifier | Epoch | Train Loss | Val Loss | AUROC | AUPRC | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MLP | 1 | 0.6728 | 0.5955 | 0.7779 | 0.7585 | 0.7414 |
| MLP | 2 | 0.5798 | 0.5586 | 0.8084 | 0.8002 | 0.6966 |
| MLP | 3 | 0.5140 | 0.5617 | 0.8215 | 0.8201 | 0.7638 |
| CNN | 1 | 0.7346 | 0.7056 | 0.5196 | 0.5241 | 0.2857 |
| CNN | 2 | 0.7166 | 0.6998 | 0.5533 | 0.5530 | 0.1404 |
| CNN | 3 | 0.7052 | 0.6811 | 0.6461 | 0.6373 | 0.3037 |
| Gated | 1 | 0.6438 | 0.5592 | 0.7890 | 0.7750 | 0.7483 |
| Gated | 2 | 0.5748 | 0.5457 | 0.8054 | 0.7937 | 0.7143 |
| Gated | 3 | 0.5476 | 0.5228 | 0.8174 | 0.8072 | 0.7729 |

## Best Epoch By Classifier

| Classifier | Best Epoch | Selection Basis | Val Loss | AUROC | AUPRC | F1 | Checkpoint |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| MLP | 3 | highest AUPRC | 0.5617 | 0.8215 | 0.8201 | 0.7638 | `model/esm2_bilstm_mlp_tune_reg_h256_p128_d045_lr5e5_3k.pt` |
| CNN | 3 | highest AUPRC | 0.6811 | 0.6461 | 0.6373 | 0.3037 | `model/esm2_bilstm_cnn_tune_cnn_h256_p128_d045_lr5e5_3k.pt` |
| Gated | 3 | highest AUPRC | 0.5228 | 0.8174 | 0.8072 | 0.7729 | `model/esm2_bilstm_gated_tune_gated_h256_p128_d045_lr5e5_3k.pt` |

## Recommendation

Recommend `MLPClassifier` for the next grid search and full-data training.

Rationale:
- It achieved the best validation AUPRC, `0.8201`, which matches the current checkpoint selection criterion and is the most relevant metric for balanced disease/pathogenicity ranking.
- It also had the best AUROC, `0.8215`, while staying close to the best F1.
- It is simpler than the gated head and much stronger than the CNN head, so it is the lowest-risk choice for full-data training.

The `GatedResidualClassifier` is the best backup candidate. It had the lowest validation loss, `0.5228`, and best F1, `0.7729`, but its AUPRC was lower than MLP. If later calibration or fixed-threshold classification becomes more important than ranking, include `gated` in a secondary search.

Do not prioritize the CNN head for the next round. It improved by epoch 3 but remained far below the dense heads, suggesting the current pair feature is better handled as dense interactions than as local convolutional structure.

## Suggested Next Grid

Use `CLASSIFIER_TYPE="mlp"` in `script/train.py` and search:

| Parameter | Values |
| --- | --- |
| `LR` | `3e-5`, `5e-5`, `8e-5` |
| `DROPOUT` | `0.35`, `0.45`, `0.55` |
| `PROJ_DIM` | `128`, `192` |
| `LSTM_HIDDEN` | `256` |
| `EPOCHS` | `3`, with early stopping monitored from epoch 2 onward |

Current recommended full-run starting point:

```python
CLASSIFIER_TYPE = "mlp"
TAG = "reg_h256_p128_d045_lr5e5"
EPOCHS = 3
BATCH_SIZE = 2
LR = 5e-5
LSTM_HIDDEN = 256
PROJ_DIM = 128
DROPOUT = 0.45
```

