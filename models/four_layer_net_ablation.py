import torch
import torch.nn as nn
import torch.nn.functional as F

class FourLayerAblation(nn.Module):
    """
    具有4個有效層的網絡，但移除注意力機制，用於消融實驗
    
    有效層定義:
    1. 初始特徵提取層：大型卷積
    2. 中間卷積層 1（代替注意力輸入投影層）
    3. 中間卷積層 2（代替注意力輸出投影層）
    4. 分類層
    """
    def __init__(self, num_classes=50):
        """
        初始化 FourLayerAblation
        
        Args:
            num_classes (int, optional): 分類類別數量
        """
        super(FourLayerAblation, self).__init__()
        
        # 有效層 1：初始特徵提取層 - 使用大型卷積快速擴大感受野（與原模型相同）
        self.conv_layer = nn.Conv2d(3, 256, kernel_size=11, stride=4, padding=5)
        
        # 輔助操作（不計入有效層）
        self.bn1 = nn.BatchNorm2d(256)
        self.silu = nn.SiLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # 有效層 2：中間卷積層 1（代替注意力輸入投影層）
        # 使用3x3卷積保持空間信息，同時增加通道數
        self.mid_conv1 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        
        # 為中間卷積層1添加專用的批標準化層
        self.bn_mid = nn.BatchNorm2d(384)
        
        # 有效層 3：中間卷積層 2（代替注意力輸出投影層）
        self.mid_conv2 = nn.Conv2d(384, 512, kernel_size=3, padding=1)
        
        # 輔助操作（不計入有效層）
        self.bn2 = nn.BatchNorm2d(512)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 有效層 4：分類層（與原模型相同）
        self.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        """
        前向傳播
        
        Args:
            x (torch.Tensor): 輸入張量，形狀為 (batch_size, 3, height, width)
            
        Returns:
            torch.Tensor: 輸出張量，形狀為 (batch_size, num_classes)
        """
        # 第一層：初始特徵提取
        x = self.conv_layer(x)  # 有效層 1
        x = self.bn1(x)
        x = self.silu(x)
        x = self.pool(x)
        
        # 第二層：中間卷積層 1（替代注意力輸入投影）
        x = self.mid_conv1(x)  # 有效層 2
        x = self.bn_mid(x)  # 使用專用的BatchNorm
        x = self.silu(x)
        
        # 第三層：中間卷積層 2（替代注意力輸出投影）
        x = self.mid_conv2(x)  # 有效層 3
        x = self.bn2(x)
        x = self.silu(x)
        
        # 全局池化準備分類
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # 第四層：分類層
        x = self.fc(x)  # 有效層 4
        
        return x