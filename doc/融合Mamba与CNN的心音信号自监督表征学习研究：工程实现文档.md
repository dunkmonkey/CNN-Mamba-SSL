

# **融合Mamba与CNN的心音信号自监督表征学习研究：工程实现文档**

本文档提供了一个详细的工程实现指南，用于构建、训练和应用一个基于自监督学习的串行混合编码器（1D-CNN \+ Mamba），以进行心音信号的表征学习。

## **1\. 环境设置**

首先，确保您的Python环境中已安装必要的库。建议使用conda创建独立的虚拟环境。

Bash

\# 1\. 创建并激活Conda环境  
conda create \-n heart\_ssl python=3.10 \-y  
conda activate heart\_ssl

\# 2\. 安装PyTorch (请根据您的CUDA版本从PyTorch官网选择合适的命令)  
\# 示例 (CUDA 11.8):  
pip install torch torchvision torchaudio \--index-url https://download.pytorch.org/whl/cu118

\# 3\. 安装Mamba及相关依赖  
pip install mamba\_ssm causal\_conv1d

\# 4\. 安装其他常用库  
pip install numpy pandas scikit-learn librosa

## **2\. 混合编码器架构实现**

我们将使用PyTorch的nn.Module来定义串行的CNN-Mamba混合编码器。该编码器接收一维心音信号，并输出一个固定维度的特征向量。

Python

import torch  
import torch.nn as nn  
import torch.nn.functional as F  
from mamba\_ssm import Mamba

class HybridEncoder(nn.Module):  
    """  
    一个串行的1D-CNN与Mamba混合编码器，用于心音信号表征学习。  
    \- CNN前端用于提取局部特征并降采样。  
    \- Mamba后端用于捕捉序列中的长程依赖关系。  
    """  
    def \_\_init\_\_(self, cnn\_out\_channels=128, mamba\_d\_state=16, mamba\_d\_conv=4, mamba\_expand=2):  
        super().\_\_init\_\_()  
          
        \# 1\. 1D-CNN 前端  
        \# 包含多个卷积层，用于逐步提取特征并降低序列长度  
        self.cnn\_frontend \= nn.Sequential(  
            nn.Conv1d(in\_channels=1, out\_channels=32, kernel\_size=16, stride=2, padding=7),  
            nn.BatchNorm1d(32),  
            nn.ReLU(),  
            nn.MaxPool1d(kernel\_size=2, stride=2),  
              
            nn.Conv1d(in\_channels=32, out\_channels=64, kernel\_size=8, stride=2, padding=3),  
            nn.BatchNorm1d(64),  
            nn.ReLU(),  
            nn.MaxPool1d(kernel\_size=2, stride=2),

            nn.Conv1d(in\_channels=64, out\_channels=cnn\_out\_channels, kernel\_size=4, stride=2, padding=1),  
            nn.BatchNorm1d(cnn\_out\_channels),  
            nn.ReLU()  
        )  
          
        \# 2\. Mamba 后端  
        \# Mamba的输入维度(d\_model)必须与CNN的输出通道数(cnn\_out\_channels)一致  
        self.mamba\_backend \= Mamba(  
            d\_model=cnn\_out\_channels,  
            d\_state=mamba\_d\_state,  
            d\_conv=mamba\_d\_conv,  
            expand=mamba\_expand,  
        )  
          
    def forward(self, x: torch.Tensor) \-\> torch.Tensor:  
        """  
        前向传播函数  
        Args:  
            x (torch.Tensor): 输入的原始心音信号，形状为 (batch\_size, 1, signal\_length)  
        Returns:  
            torch.Tensor: 输出的固定维度特征向量，形状为 (batch\_size, cnn\_out\_channels)  
        """  
        \# 1\. 通过CNN前端提取局部特征  
        \# 输入形状: (B, 1, L\_in) \-\> 输出形状: (B, C\_out, L\_out)  
        local\_features \= self.cnn\_frontend(x)  
          
        \# 2\. 重塑以适应Mamba的输入格式  
        \# Mamba期望的输入形状为 (B, L, D)  
        \# (B, C\_out, L\_out) \-\> (B, L\_out, C\_out)  
        local\_features \= local\_features.transpose(1, 2)  
          
        \# 3\. 通过Mamba后端捕捉长程依赖  
        \# 输入形状: (B, L\_out, C\_out) \-\> 输出形状: (B, L\_out, C\_out)  
        global\_representation \= self.mamba\_backend(local\_features)  
          
        \# 4\. 使用平均池化得到整个序列的固定大小向量表示  
        \# (B, L\_out, C\_out) \-\> (B, C\_out)  
        final\_vector \= global\_representation.mean(dim=1)  
          
        return final\_vector

## **3\. MoCo框架集成与预训练流程**

接下来，我们将HybridEncoder集成到MoCo框架中进行自监督预训练。

### **3.1 模型和队列初始化**

在训练脚本的开始部分，需要初始化查询网络、键网络、投影头以及负样本队列。

Python

\# \--- 模型参数 \---  
CNN\_OUT\_CHANNELS \= 128  
PROJECTION\_DIM \= 128  
QUEUE\_SIZE \= 65536  
MOMENTUM \= 0.999  
TEMPERATURE \= 0.07

\# \--- 初始化编码器 \---  
encoder\_q \= HybridEncoder(cnn\_out\_channels=CNN\_OUT\_CHANNELS)  
encoder\_k \= HybridEncoder(cnn\_out\_channels=CNN\_OUT\_CHANNELS)

\# \--- 初始化投影头 (MLP) \---  
\# MoCo v2的改进，有助于学习更好的表征  
projector\_q \= nn.Sequential(nn.Linear(CNN\_OUT\_CHANNELS, CNN\_OUT\_CHANNELS), nn.ReLU(), nn.Linear(CNN\_OUT\_CHANNELS, PROJECTION\_DIM))  
projector\_k \= nn.Sequential(nn.Linear(CNN\_OUT\_CHANNELS, CNN\_OUT\_CHANNELS), nn.ReLU(), nn.Linear(CNN\_OUT\_CHANNELS, PROJECTION\_DIM))

\# \--- 初始化键网络权重 \---  
\# 确保键网络的初始权重与查询网络完全一致  
for param\_q, param\_k in zip(encoder\_q.parameters(), encoder\_k.parameters()):  
    param\_k.data.copy\_(param\_q.data)  
    param\_k.requires\_grad \= False \# 键网络不通过反向传播更新

for param\_q, param\_k in zip(projector\_q.parameters(), projector\_k.parameters()):  
    param\_k.data.copy\_(param\_q.data)  
    param\_k.requires\_grad \= False

\# \--- 初始化负样本队列 \---  
\# 注册为buffer，这样它会随模型移动（例如 to(device)），但不是模型参数  
\# self.register\_buffer("queue", torch.randn(PROJECTION\_DIM, QUEUE\_SIZE))  
\# self.register\_buffer("queue\_ptr", torch.zeros(1, dtype=torch.long))

### **3.2 训练循环详解 (伪代码)**

以下是单个训练批次的伪代码，展示了MoCo的核心逻辑。

Python

\# optimizer只包含查询网络(encoder\_q, projector\_q)的参数  
optimizer \= torch.optim.Adam(list(encoder\_q.parameters()) \+ list(projector\_q.parameters()), lr=1e-3)

for batch in dataloader: \# dataloader提供无标签的心音信号  
    \# (a) 数据加载与增强  
    x \= batch \# 形状: (B, 1, L)  
    x\_q \= augmentation\_pipeline\_1(x) \# 应用第一种随机增强  
    x\_k \= augmentation\_pipeline\_2(x) \# 应用第二种随机增强

    \# (b) 编码查询视图  
    q\_features \= encoder\_q(x\_q)  
    q \= projector\_q(q\_features)  
    q \= F.normalize(q, dim=1) \# L2归一化

    \# (c) 编码键视图 (无梯度计算)  
    with torch.no\_grad():  
        k\_features \= encoder\_k(x\_k)  
        k \= projector\_k(k\_features)  
        k \= F.normalize(k, dim=1)

    \# (d) 计算对比损失 (InfoNCE)  
    \# 正样本相似度  
    l\_pos \= torch.einsum('nc,nc-\>n', \[q, k\]).unsqueeze(-1)  
    \# 负样本相似度  
    l\_neg \= torch.einsum('nc,ck-\>nk', \[q, queue.clone().detach()\])  
      
    logits \= torch.cat(\[l\_pos, l\_neg\], dim=1) / TEMPERATURE  
    labels \= torch.zeros(logits.shape, dtype=torch.long).to(device)  
      
    loss \= F.cross\_entropy(logits, labels)

    \# (e) 反向传播与优化  
    optimizer.zero\_grad()  
    loss.backward()  
    optimizer.step()

    \# (f) 动量更新键网络  
    with torch.no\_grad():  
        for param\_q, param\_k in zip(encoder\_q.parameters(), encoder\_k.parameters()):  
            param\_k.data \= param\_k.data \* MOMENTUM \+ param\_q.data \* (1\. \- MOMENTUM)  
        for param\_q, param\_k in zip(projector\_q.parameters(), projector\_k.parameters()):  
            param\_k.data \= param\_k.data \* MOMENTUM \+ param\_q.data \* (1\. \- MOMENTUM)

    \# (g) 队列管理 (入队与出队)  
    \# \_dequeue\_and\_enqueue(k)

## **4\. 下游任务微调**

当自监督预训练完成后，encoder\_q就成为了一个强大的、通用的心音特征提取器。可以将其应用于下游的有监督分类任务。

### **4.1 模型准备**

加载预训练好的查询编码器encoder\_q的权重，并丢弃键编码器和投影头。

### **4.2 添加分类头**

在预训练的编码器之上，添加一个简单的线性分类头。

Python

class DownstreamClassifier(nn.Module):  
    def \_\_init\_\_(self, pretrained\_encoder, num\_classes):  
        super().\_\_init\_\_()  
        self.encoder \= pretrained\_encoder  
          
        \# 获取编码器的输出维度  
        encoder\_output\_dim \= self.encoder.mamba\_backend.d\_model  
          
        self.classifier\_head \= nn.Linear(encoder\_output\_dim, num\_classes)

    def forward(self, x):  
        \# 可以选择冻结编码器  
        with torch.no\_grad():  
            features \= self.encoder(x)  
          
        \# 或者只训练分类头  
        \# features \= self.encoder(x)  
          
        logits \= self.classifier\_head(features)  
        return logits

\# 加载预训练权重  
pretrained\_encoder \= HybridEncoder(cnn\_out\_channels=CNN\_OUT\_CHANNELS)  
pretrained\_encoder.load\_state\_dict(torch.load("pretrained\_encoder\_q.pth"))

\# 创建分类模型  
model \= DownstreamClassifier(pretrained\_encoder, num\_classes=5) \# 假设有5个类别

### **4.3 训练策略**

使用**少量有标签**的心音数据对DownstreamClassifier进行训练。常见的策略有：

1. **线性评估（Linear Probing）：** 完全冻结encoder的权重，只训练classifier\_head。这可以快速评估预训练表征的质量。  
2. **全网络微调（Fine-tuning）：** 训练整个网络，但为encoder设置一个比classifier\_head小得多的学习率（例如，小10-100倍），以避免破坏预训练学到的宝贵特征。

通过以上步骤，您可以实现一个完整的、基于CNN-Mamba混合架构的自监督学习流程，为心音信号分析任务构建一个强大的基础模型。