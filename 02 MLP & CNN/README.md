# Fashion-MNIST 图像分类：MLP 与 CNN

本项目基于 PyTorch 实现了对 Fashion-MNIST 数据集的图像分类任务。作为课程作业，代码主要注重从零搭建模型的框架，并且对比了多层感知机（MLP）与卷积神经网络（CNN）在该数据集上的表现。

## 数据准备 (Data Preparation)

运行本项目代码前，请确保 `data` 文件夹准备就绪。你有两种方式准备数据：

**方式一：修改代码自动下载（推荐）**
找到代码中加载 `FashionMNIST` 的部分，将 `download=False` 改为 `download=True`。首次运行时会自动下载并解压数据集到本地 `data` 目录。

**方式二：手动下载（代码默认方式）**
本项目代码默认设置为 `download=False`。请在当前目录下创建一个名为 `data` 的文件夹，并从 [Fashion-MNIST 官网](https://github.com/zalandoresearch/fashion-mnist) 下载四个压缩包（`train-images-idx3-ubyte.gz` 等），解压至 `data/FashionMNIST/raw/` 目录下。

## 数据集
**Fashion-MNIST**：包含 10 个类别的服饰灰度图像（如 T恤、裤子、裙子、鞋子等），每张图片大小为 28x28。代码中包含数据的自动定位、标准化处理及可视化展示。

## 模型实现
*   **MLP (多层感知机)**：使用 Flatten 展开图像，包含两层全连接层（256 -> 128），并配合 ReLU 激活函数。参数量约为 235K。
*   **CNN (卷积神经网络)**：包含两个卷积块。第一层使用 32 个 3x3 卷积核，第二层增加到 64 个，均配合 ReLU 激活和 2x2 最大池化层进行下采样。最后将 64 个 7x7 的特征图展平，接入全连接层完成分类。

## 主要做了什么
*   **数据加载**：使用 `DataLoader` 进行批次划分。
*   **训练与评估**：实现了标准的 `train_one_epoch` 和 `evaluate` 函数，监控 Loss 和 Accuracy。
*   **结果可视化**：绘制了训练与测试过程中的 Loss 和 Accuracy 曲线，方便确认模型收敛情况。
*   **结果总结**：对模型训练结果进行了整理与总结，提出了自己的看法。并且附加了一个参数量大致相同情况下的性能对照实验。
