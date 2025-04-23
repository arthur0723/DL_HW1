import torch
import torch.nn as nn
import torch.nn.functional as F

class FourLayerNet(nn.Module):
    """
    具有4個有效層的網絡，用於圖像分類
    
    有效層定義:
    1. 初始特徵提取層：大型卷積
    2. 注意力輸入投影層
    3. 注意力輸出投影層
    4. 分類層
    """
    def __init__(self, num_classes=50):
        """
        初始化 FourLayerNet
        
        Args:
            num_classes (int, optional): 分類類別數量
        """
        super(FourLayerNet, self).__init__()
        
        # 有效層 1：初始特徵提取層 - 使用大型卷積快速擴大感受野
        self.conv_layer = nn.Conv2d(3, 256, kernel_size=11, stride=4, padding=5)
        
        # 輔助操作（不計入有效層）
        self.bn1 = nn.BatchNorm2d(256)
        self.silu = nn.SiLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # 計算自注意力的參數
        self.feat_size = 27  # 根據輸入尺寸和第一層操作後的特徵圖大小計算
        self.num_heads = 8
        self.head_dim = 32
        self.attn_dim = self.num_heads * self.head_dim
        
        # 有效層 2：多頭注意力的輸入投影
        # 單一大型線性層用於Q、K、V投影（這些投影在一個矩陣中實現）
        self.qkv_proj = nn.Conv2d(256, 3 * self.attn_dim, kernel_size=1)
        
        # 有效層 3：多頭注意力的輸出投影
        self.out_proj = nn.Conv2d(self.attn_dim, 512, kernel_size=1)
        
        # 輔助操作（不計入有效層）
        self.bn2 = nn.BatchNorm2d(512)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 有效層 4：分類層
        self.fc = nn.Linear(512, num_classes)
        
        # 位置編碼（不計入有效層，只是一個固定的加法操作）
        self.register_buffer("pos_embed", self._create_pos_embed())
    
    def _create_pos_embed(self):
        """
        創建固定的位置編碼（不計入有效層）
        
        Returns:
            torch.Tensor: 位置編碼張量
        """
        pos_embed = torch.zeros(1, 256, self.feat_size, self.feat_size)
        y_embed = torch.linspace(-1., 1., self.feat_size)
        x_embed = torch.linspace(-1., 1., self.feat_size)
        y_embed, x_embed = torch.meshgrid(y_embed, x_embed, indexing='ij')
        
        # 使用正弦-餘弦位置編碼的簡化版本
        pos_embed[0, 0::4, :, :] = torch.sin(x_embed * torch.pi)
        pos_embed[0, 1::4, :, :] = torch.cos(x_embed * torch.pi)
        pos_embed[0, 2::4, :, :] = torch.sin(y_embed * torch.pi)
        pos_embed[0, 3::4, :, :] = torch.cos(y_embed * torch.pi)
        
        return pos_embed
    
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
        
        # 添加位置編碼
        if x.size(2) == self.feat_size and x.size(3) == self.feat_size:
            x = x + self.pos_embed
        
        # 保存輸入的形狀以便後續重塑
        b, c, h, w = x.shape
        
        # 第二層：多頭注意力的輸入投影
        qkv = self.qkv_proj(x)  # 有效層 2
        qkv = qkv.reshape(b, 3, self.num_heads, self.head_dim, h * w)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # [3, b, num_heads, h*w, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 計算注意力分數（點積注意力，不計入有效層，僅是矩陣運算）
        attn = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim ** 0.5))
        attn = F.softmax(attn, dim=-1)
        
        # 應用注意力得到加權值
        x = (attn @ v).transpose(1, 2).reshape(b, self.attn_dim, h, w)
        
        # 第三層：多頭注意力的輸出投影
        x = self.out_proj(x)  # 有效層 3
        x = self.bn2(x)
        x = self.silu(x)
        
        # 全局池化準備分類
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # 第四層：分類層
        x = self.fc(x)  # 有效層 4
        
        return x