import torch
import torch.nn as nn
from torchvision import models

def create_resnet34(num_classes, weights=None):
    """
    創建 ResNet34 模型
    
    Args:
        num_classes (int): 分類類別數量
        weights (torchvision.models.ResNet34_Weights, optional): 預訓練權重
    
    Returns:
        nn.Module: ResNet34 模型
    """
    # 較新版本的PyTorch使用 weights=None 代替 pretrained=False
    model = models.resnet34(weights=weights)
    
    # 修改最後一層以適應數據集的類別數
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model