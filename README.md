# ResNet-Me: 深入学习图像识别

## 项目概述

**ResNet-Me** 是一个基于 PyTorch 实现的深度残差网络 (ResNet). 本项目旨在为研究人员和开发人员提供一个清晰、易于使用和修改的 ResNet 模型，用于图像分类任务。ResNet 通过引入“快捷连接”或“跳跃连接”来解决深度神经网络中的梯度消失问题，从而能够训练更深的网络并获得更高的准确率.

该代码库实现了标准的 ResNet 架构

## 主要特性

*   **多种 ResNet 模型**: 支持 ResNet-18, ResNet-34, ResNet-50, ResNet-101 和 ResNet-152 等多种经典模型.
*   **模块化设计**: 代码结构清晰，易于理解和扩展，可以方便地将 ResNet 模型集成到其他项目中.
*   **自定义配置**: 训练过程中的超参数，如学习率、批量大小、训练周期等，均可通过配置文件或命令行参数进行调整.
*   **数据增强**: 训练脚本中包含了常见的数据增强策略，如随机裁剪、翻转等，以提升模型的泛化能力.
*   **易于上手的训练脚本**: 提供了一个完整的 `train.py` 脚本，用户只需指定数据集路径和其他参数即可开始训练.

## 安装

1.  **克隆代码库**:
    ```bash
    git clone https://github.com/foorgange/resnet-me.git
    cd resnet-me
    ```

2.  **创建并激活 Conda 环境 (推荐)**:
    ```bash
    conda create -n resnet-me python=3.8
    conda activate resnet-me
    ```

3.  **安装依赖**:
    ```bash
    pip install -r requirements.txt
    ```
    *(注意: `requirements.txt` 文件需要您根据项目中的 `import` 语句自行创建，主要应包含 `torch`, `torchvision`, `numpy` 等)*

## 使用方法

### 数据集准备

请将您的图像分类数据集按照以下结构进行组织：

```
/path/to/your/dataset/
├── train/
│   ├── class1/
│   │   ├── xxx.jpg
│   │   └── ...
│   └── class2/
│       ├── yyy.jpg
│       └── ...
└── val/
    ├── class1/
    │   ├── zzz.jpg
    │   └── ...
    └── class2/
        ├── www.jpg
        └── ...
```

### 训练模型

您可以通过运行 `train.py` 脚本来训练模型。可以通过命令行参数来指定不同的配置.

**基本训练示例**:

```bash
python train.py --data_path /path/to/your/dataset --model_name resnet50 --epochs 100 --batch_size 64 --lr 0.01
```

**可配置的命令行参数**:

*   `--data_path`: 数据集所在的路径.
*   `--model_name`: 要训练的 ResNet 模型名称 (例如: `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`).
*   `--epochs`: 训练的总轮数.
*   `--batch_size`: 每个批次的图像数量.
*   `--lr`: 初始学习率.
*   `--device`: 指定训练设备 (例如: `cuda:0` 或 `cpu`).
*   以及其他在 `train.py` 中定义的参数.

### 配置文件

您也可以直接修改 `config.py` 文件来设置默认的训练参数，这对于不希望每次都在命令行中输入大量参数的用户来说非常方便.

## 文件结构

```
resnet-me/
├── resnet.py          # ResNet 模型的核心实现
├── train.py           # 训练和评估脚本
├── config.py          # 默认配置文件
├── utils.py           # 工具函数 (例如: 准确率计算、日志记录等)
└── README.md          # 本文档
```

## ResNet 架构简介

ResNet (残差网络) 的核心思想是学习残差函数. 在传统的深度网络中，每一层都试图直接学习一个目标映射 `H(x)`. 而在 ResNet 中，层被重新组织为学习一个残差函数 `F(x) = H(x) - x`，原始的映射则变为 `H(x) = F(x) + x`. 这种设计使得网络更容易学习恒等映射，从而解决了深度网络难以训练的问题.

ResNet 架构在许多计算机视觉任务中都取得了卓越的成果，包括图像分类、目标检测和图像分割等.

##
