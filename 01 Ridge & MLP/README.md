# Kaggle 房价预测：Ridge 与 MLP

本项目基于 PyTorch 实现了对 Kaggle 经典竞赛（House Prices - Advanced Regression Techniques）的房价预测任务。作为课程作业，代码主要注重从零搭建模型的框架，并且对比了岭回归（Ridge）与多层感知机（MLP）在该数据集上的表现。

## 数据准备 (Data Preparation)

运行本项目代码前，请确保 `data` 文件夹准备就绪。

请在代码所在目录下创建一个名为 `data` 的文件夹，并在其中创建 `house_prices` 子文件夹。从 Kaggle 官网下载 `train.csv` 和 `test.csv`，并放入 `data/house_prices/` 目录下。
*(注：代码中包含自动向上查找 `data` 文件夹的逻辑，直接运行即可定位到数据)*

## 数据集预处理

**Kaggle 房价数据集**：包含约 1460 条训练数据和 1459 条测试数据，涵盖了房屋面积、年份、质量等 79 个特征。
代码中对数据进行了较为细致的清洗与特征工程：
- 剔除了不合理的异常值（如面积过大但价格极低的样本）。
- 对目标房价（SalePrice）进行对数转换，缓解右偏分布。
- 对缺失值进行分类处理（数值型取中位数，类别型填 "None"）。
- 对类别特征进行独热编码（One-Hot Encoding），对偏态数值特征取对数。
- 使用 `StandardScaler` 对全部特征进行标准化。

## 模型实现
**Ridge (岭回归)**：使用 `nn.Linear` 实现的最简单线性层，在优化器中通过设置较高的 `weight_decay=1.0` 来实现 L2 正则化，防止过拟合。
**MLP (多层感知机)**：包含两个隐藏层的全连接网络（输入层 -> 64 -> 32 -> 1），激活函数采用 ReLU，配合 `weight_decay=0.1` 进行训练。

## 主要做了什么
**数据切分**：按 8:2 的比例手动切分了训练集与验证集。
**训练与评估**：实现了标准的 `train_one_epoch` 和 `evaluate` 函数，采用 MSELoss 监控训练与验证的 RMSE（均方根误差）。
**结果可视化**：绘制了 500 个 epoch 的 Loss 曲线，方便确认模型是否收敛。最终 Ridge 验证集 RMSE 约为 0.303，MLP 约为 0.308。
