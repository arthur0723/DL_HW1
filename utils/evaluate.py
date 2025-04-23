import os
import torch
import torch.nn as nn
import numpy as np
import json
from thop import profile
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support

def evaluate_model(model, dataloader, device='cuda', dataset_name='測試集'):
    """
    評估模型性能
    
    Args:
        model (nn.Module): 要評估的模型
        dataloader (DataLoader): 數據加載器
        device (str, optional): 設備 ('cuda' 或 'cpu')
        dataset_name (str, optional): 數據集名稱，用於顯示
    
    Returns:
        tuple: (acc, all_preds, all_labels) - 準確率、所有預測和所有標籤
    """
    print(f'在{dataset_name}上評估模型...')
    model.eval()
    running_corrects = 0
    all_preds = []
    all_labels = []
    
    # 遍歷數據
    total_batches = len(dataloader)
    batch_count = 0
    
    for inputs, labels in dataloader:
        batch_count += 1
        print(f'  評估批次 [{batch_count}/{total_batches}]', end='\r')
        
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # 前向傳播
        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
        # 儲存預測和標籤以計算指標
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # 統計正確預測數量
        running_corrects += torch.sum(preds == labels.data)
    
    # 計算準確率
    acc = running_corrects.double() / len(dataloader.dataset)
    print(f'\n{dataset_name}準確率: {acc:.4f}')
    
    return acc.item(), all_preds, all_labels

def calculate_metrics(all_preds, all_labels, class_names=None):
    """
    計算性能指標（准確率、精確度、召回率、F1分數）
    
    Args:
        all_preds (list): 所有預測
        all_labels (list): 所有標籤
        class_names (list, optional): 類別名稱列表
    
    Returns:
        dict: 包含各種性能指標的字典
    """
    # 計算混淆矩陣
    cm = confusion_matrix(all_labels, all_preds)
    
    # 計算精確度、召回率、F1分數
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
    
    # 計算總體准確率
    accuracy = np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    
    # 創建性能指標字典
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm.tolist()
    }
    
    return metrics

def calculate_params_and_flops(model, input_size=(3, 224, 224), batch_size=1):
    """
    計算模型的參數數量和計算複雜度
    
    Args:
        model (nn.Module): 要分析的模型
        input_size (tuple, optional): 輸入張量的大小 (不包括批次大小)
        batch_size (int, optional): 批次大小
    
    Returns:
        tuple: (params, flops) - 參數數量和浮點運算數
    """
    # 確保模型處於評估模式
    model.eval()
    
    # 創建輸入張量
    input_tensor = torch.randn(batch_size, *input_size).to(next(model.parameters()).device)
    
    # 計算FLOPs和參數
    try:
        macs, params = profile(model, inputs=(input_tensor,))
        flops = macs * 2  # 1 MAC = 2 FLOPs
    except:
        # 如果thop模塊不可用，手動計算參數數量，但無法計算FLOPs
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        flops = None
        print("警告: 無法使用thop模塊計算FLOPs，僅計算參數數量")
    
    print(f"模型參數數量: {params:,}")
    if flops is not None:
        print(f"模型計算複雜度 (FLOPs): {flops:,}")
    
    return params, flops

def plot_confusion_matrix(cm, save_path, class_names=None, normalize=True):
    """
    繪製混淆矩陣
    
    Args:
        cm (numpy.ndarray): 混淆矩陣
        save_path (str): 圖表保存路徑
        class_names (list, optional): 類別名稱列表
        normalize (bool, optional): 是否歸一化混淆矩陣
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    if class_names is not None:
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
    
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    plt.close()

def save_evaluation_results(results, metrics, save_dir):
    """
    保存評估結果
    
    Args:
        results (dict): 評估結果字典
        metrics (dict): 性能指標字典
        save_dir (str): 保存目錄
    """
    # 將所有結果合併到一個字典中
    all_results = {**results, **metrics}
    
    # 保存為JSON文件
    with open(os.path.join(save_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(all_results, f, indent=4)
    
    # 繪製混淆矩陣
    if 'confusion_matrix' in metrics:
        cm = np.array(metrics['confusion_matrix'])
        plot_confusion_matrix(cm, os.path.join(save_dir, 'confusion_matrix.png'))
    
    print(f"評估結果已保存至 {os.path.join(save_dir, 'evaluation_results.json')}")