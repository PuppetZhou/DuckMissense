你是一位大模型训练师，现在我正在训练一个深度学习二分类模型，预测蛋白质突变致病性
你需要监督模型的训练情况，根据 val loss 和 其他指标进行调整，最终得到一个合适的模型架构，以及该架构下的超参数。

你的权限：
- 你可以微调模型/home/xuyzh/d2l/SYSU-hw/MutDisease/script/model.py，但是仅限于：
    - 增加 drop，或者增加LayerNorm / BatchNorm。
    - 可以对下游分类器进行优化，当前为 MLP
- 你可以对训练方式进行优化，即超参数，这是你确定好模型架构后的最重要的优化目标：
    - 你可以设置网格搜索探索超参数
    - 不要对当前的训练脚本进行其他修改，除非记录一下指标 log 方便你查看
    - 你可以优化/home/xuyzh/d2l/SYSU-hw/MutDisease/script/prepare_data.py数据集划分比例，同时需要特别注意由于训练时间较长，当前 8000 条序列的一个 epoch 需要30min，因此选择合适大小的子集进行训练。全量数据为 30 万。

当前的问题：
- 目前使用脚本的初始参数进行 train，会遇到严重的过拟合问题（数据样本比例划分3:1），val loss 在第三个 epoch 就开始上升，train loss 持续性下降
- 我已经重新划分训练集：val: n=12569 pos=6285 neg=6284，test: n=12569 pos=6284 neg=6285

训练时记得使用 tag：在/home/xuyzh/d2l/SYSU-hw/MutDisease/script/train.py脚本中的 TAG 参数进行修改




