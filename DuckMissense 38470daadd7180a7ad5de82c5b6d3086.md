# DuckMissense

## 项目背景

先简单介绍 missense variant

<aside>
💡

错义突变（missense variant）是指基因组中单个核苷酸改变导致蛋白质中一个氨基酸被替换的变异。这种改变可能破坏蛋白质的结构稳定性、酶活性或相互作用，从而与遗传性疾病、肿瘤发生、药物反应等密切相关。临床上，错义突变是最常见且最难解读的变异类型之一。

</aside>

![Screenshot 2026-06-20 at 00.15.35.png](DuckMissense/Screenshot_2026-06-20_at_00.15.35.png)

再简单介绍当前的方法：

<aside>
💡

当前主流的 missense 预测方法可以分为3类：

- 基于进化保守性的方法，利用多序列比对 MSA 计算进化保守度从而预测突变效应，代表工具SIFT、PolyPhen-2 和 PROVEAN
- 基于深度生成模型的无监督方法，代表性工具包括 EVE（Evolutionary Model of Variant Effect），利用蛋白质序列的进化分布训练深度生成模型，无需人工标注的致病性数据即可估计变异效应。
- 结构信息与语言模型融合的方法，DeepMind 开发的 AlphaMissense，基于 AlphaFold 的蛋白质结构预测网络进行微调，结合人类与灵长类种群变异频率数据，利用结构上下文信息预测错义突变的致病性。
</aside>

近年来兴起的蛋白质语言模型（Protein Language Model, PLM）方法。受自然语言处理领域 Transformer 架构成功的启发，研究者将蛋白质序列视为"生物语言"，利用自监督预训练策略在大规模蛋白质序列语料上学习氨基酸的上下文表示

PLM 在蛋白质结构预测、功能注释、突变效应评估等任务中展现出强大的泛化能力，特别是 ESM 系列家族

本项目旨在构建一个高质量的突变效应注释数据集，并以此为基础，探索以蛋白质大语言模型（Protein Language Model, PLM）为基准的有监督学习策略，系统解决错义突变致病性预测中的关键科学问题。

![image.png](DuckMissense/image.png)

## 数据处理

为了构建一个高质量的 missense variant 数据集，我们选取了 3 个常用的数据库：（需要你搜索一下来源和背景，简单介绍）

- ClinVar（germline，遗传突变）
- Cosmic（癌症）
- Humsavar

对于三个不同来源的数据库我们进行了如下的质量控制与筛选：

1. 选取 missense 突变类型的条目，即单氨基酸替换
2. 按照不同数据集的质量评分进行筛选，
    1. 例如 ClinVar 数据集选择 ReviewStatus 2 star 以上的条目得到高质量准确条目
    2. 对癌症来源数据选择已经确定的癌症驱动突变条目
3. 随后对三个来源的数据集进行统一的格式规范和标签确定，去重。

最终数据集如下：（需要你稍微展示表格）

| **gene_symbol** | **ClinicalSig** | **protein_variant** | **source** | **database_id** | **mapping_id** |
| --- | --- | --- | --- | --- | --- |
| HFE | 0 (良性) | V53M | ClinVar | 15054 | NM_000410.4 |
| HFE | 0 (良性) | V59M | ClinVar | 15055 | NM_000410.4 |
| HFE | 1 (致病) | Q283P | ClinVar | 15058 | NM_000410.4 |
| CDKL5 | 1 (致病) | S603F | COSMIC | COSM1559879 | ENST00000379989.3 |
| MYLK | 1 (致病) | A1099T | COSMIC | COSM1037351 | ENST00000360304.3 |
| ARID1A | 1 (致病) | R2236P | COSMIC | COSM907760 | ENST00000324856.7 |
| A1BG | 0 (良性) | H52R | humsavar | VAR_018369 | P04217 |
| A1BG | 0 (良性) | H395R | humsavar | VAR_018370 | P04217 |
| A1CF | 0 (良性) | V555M | humsavar | VAR_052201 | Q9NQ94 |

共 **418,251 条记录**（良性 354,838 条，致病 63,413 条），分别来自 ClinVar（341,130 条）、humsavar（72,155 条）和 COSMIC（4,966 条）

## 模型架构

```bash
[B, L, 1280]          ESM2 输出
       ↓ concat AAIndex [B, L, 6]
[B, L, 1286]          拼接特征
       ↓ BiLSTM (H=128)
[B, L, 256]           双向 LSTM
       ↓ 取 mut_idx
[B, 256]              突变位点向量
       ↓ reduce (→ P=64)
[B, 64]               ref / alt 单路向量
       ↓ 4 路拼接
[B, 256]              pair_feature
       ↓ MLP
[B, 1]                logit
       ↓ squeeze
[B]                   输出

       ↓ sigmoid
[B]                   致病概率
       ↓ threshold（搜索得到）
[B]                   0/1 预测
```

BiLSTM encoder : 4,308,992
Encoder LayerNorm : 1,024
Reduce projection : 32,960
Pair LayerNorm : 512
MLP classifier : 19,073

Non-ESM total       : 4,362,561

先使用 ESM2 进行 emb

并与 AAindex 的人工数据库维度进行合并，包含 aa 的理化性质

随后使用双向 LSTM 进行阅读，从左往右，从右往左分别得到 128 维信息随后叠加

随后取出突变位点以及参考位点的向量：ALT and REF [B,256]

随后使用线性连接层进行降维至 64

随后融合 alt 和 ref 维度 得到 [B,64x4]

```python
ref_vec = [B, P]
alt_vec = [B, P]

pair_feature = torch.cat(
    [ref_vec, alt_vec, alt_vec - ref_vec, torch.abs(alt_vec - ref_vec)],
    dim=-1,
)   # [B, 4P]
```

将这个向量输入下游分类器

选择选择了一个两个隐藏层的 MLP

## 训练方法

为了兼容训练速度和正负样本平衡

选择抽样平衡正负样本子集进行模型架构调整和超参数优化

batch size ：32

loss：使用二分类交叉熵

```python
criterion = nn.BCEWithLogitsLoss()

对 logit 进行压缩至 0~1 之间，得到概率

随后使用Binary Cross Entropy：计算预测概率和真实标签之间的差距

loss = -[ y * log(p) + (1 - y) * log(1 - p) ] 
```

不需要手动 sigmoid 再传给分类器

使用 AdamW 优化器

- 每个参数有自己的自适应学习率，利用momentum
- 使用 解耦weight decay 正则化减少过拟合水平

```python
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay)
```

混合精度训练

threshold = 0.5

## 优化方法

问题：

- 训练时间较慢
- epoch 轮次增加后，出现过拟合现象
- 超参数探索

### 下游分类器

比较了三种下游分类器：

1. **MLPClassifier**
    - 两层 MLP
    - 使用 LayerNorm、GELU 和 Dropout
    - 作为主要 baseline 分类器
2. **CNNPairClassifier**
    - 将 `[ref, alt, diff, absdiff]` 视作 4 个通道
    - 使用 1D CNN 建模局部维度交互
    - 试图捕捉 pair feature 内部的局部模式
3. **GatedResidualClassifier**
    - 使用 gated dense interaction
    - 通过 gate 控制特征流，并加入 residual projection
    - 目标是增强 ref/alt 差异特征的非线性交互能力

使用了一个 3000 平衡数据集

共享训练参数：

```bash
train: 2400
val: 300
test: 300
正负样本比例: 1:1

LSTM_HIDDEN = 256
PROJ_DIM = 128
DROPOUT = 0.45
LR = 5e-5
BATCH_SIZE = 2
EPOCHS = 3
```

对比结果：

| Classifier | Best Epoch | Val Loss | AUROC | AUPRC | F1 |
| --- | --- | --- | --- | --- | --- |
| MLP | 3 | 0.5617 | 0.8215 | 0.8201 | 0.7638 |
| CNN | 3 | 0.6811 | 0.6461 | 0.6373 | 0.3037 |
| Gated | 3 | 0.5228 | 0.8174 | 0.8072 | 0.7729 |

 最终选择 MLP 作为下游分类器

同时我们认为蛋白质大语言模型的表征能力过强可能会导致过拟合现象出现

因此我们降低了蛋白质窗口切割维度，改用 ESM2 代替 ESMC

采用较为简单的浅层下游分类器

并加大正则化和 dropout 水平

### 超参数优化

使用了Optuna 贝叶斯优化搜索 7 个关键参数

采用 Optuna 贝叶斯优化自动搜索 LSTM 维度、投影维度、Dropout、学

习率、weight decay、label smoothing 及分类阈值等关键超参数

### 采样

使用平衡采样

同时考虑了同一个 gene 的突变条目出现最大次数，保证抽样后的子集中不会有大量来自同一个 gene 的蛋白

### 超参数优化

| `EPOCHS` | `15` | 最大训练轮数 |
| --- | --- | --- |
| `BATCH_SIZE` | `32` | 每批样本数 |
| `LR` | `6.4813e-5` | 学习率，介于 1e-5 ~ 1e-4 之间 |
| `WEIGHT_DECAY` | `0.005135` | AdamW 的 L2 正则强度 |
| `LSTM_HIDDEN BiLSTM 维度` | `256` | BiLSTM 单向隐层维度（输出 512） |
| `PROJ_DIM 突变位点向量投影维度` | `64` | 突变位点向量投影到 64 维 |
| `DROPOUT` | `0.5383` | 较高的 Dropout，用于抑制过拟合 |

## 测试结果

生物学实例验证

## 总结反思

对原因的分析

后续策略

- 训练集增大
- 增加维度信息
- 优化架构