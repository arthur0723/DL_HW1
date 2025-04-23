import time
import copy
import os
import torch
import matplotlib.pyplot as plt

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler=None, 
                num_epochs=10, device='cuda'):
    """
    訓練模型
    
    Args:
        model (nn.Module): 要訓練的模型
        dataloaders (dict): 包含訓練和驗證數據加載器的字典
        dataset_sizes (dict): 包含訓練和驗證數據集大小的字典
        criterion (nn.Module): 損失函數
        optimizer (torch.optim.Optimizer): 優化器
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): 學習率調度器
        num_epochs (int, optional): 訓練的周期數
        device (str, optional): 訓練設備 ('cuda' 或 'cpu')
    
    Returns:
        tuple: (best_model, history) - 最佳模型和訓練歷史
    """
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # 記錄訓練過程中的損失和準確率
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 40)
        
        # 每個 epoch 都有訓練和驗證階段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 設置模型為訓練模式
            else:
                model.eval()   # 設置模型為評估模式
                
            running_loss = 0.0
            running_corrects = 0
            
            # 用於計算批次進度
            batch_count = 0
            total_batches = len(dataloaders[phase])
            
            # 遍歷數據
            for inputs, labels in dataloaders[phase]:
                batch_start = time.time()
                
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # 將參數梯度歸零
                optimizer.zero_grad()
                
                # 前向傳播
                # 只有在訓練時才跟踪歷史記錄
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # 如果是訓練階段，則反向傳播 + 優化
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                # 統計
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                # 計算並顯示批次進度
                batch_count += 1
                batch_time = time.time() - batch_start
                
                # 每 10 個批次或最後一個批次顯示進度
                if batch_count % 10 == 0 or batch_count == total_batches:
                    current_loss = running_loss / (batch_count * inputs.size(0))
                    current_acc = running_corrects.double() / (batch_count * inputs.size(0))
                    print(f'  {phase} Batch [{batch_count}/{total_batches}] - '
                          f'Loss: {current_loss:.4f} Acc: {current_acc:.4f} '
                          f'Time: {batch_time:.2f}s')
            
            # 每個 epoch 結束後計算損失和準確率
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            print(f'{phase} Epoch Summary - Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # 記錄歷史
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
                
                # 對於 ReduceLROnPlateau 調度器，需要在驗證階段進行步進
                if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(epoch_loss)
            
            # 如果是驗證階段且性能更好，則保存模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print(f'  ** 新的最佳模型! 準確率: {best_acc:.4f} **')
        
        # 對於其他調度器，在每個 epoch 結束後進行步進
        if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step()
                
        # 計算並顯示每個 epoch 的時間
        epoch_time = time.time() - epoch_start
        print(f'Epoch 完成，耗時: {epoch_time // 60:.0f}m {epoch_time % 60:.0f}s')
        print(f'目前最佳驗證準確率: {best_acc:.4f}')
        print('-' * 40)
        
    time_elapsed = time.time() - since
    print(f'訓練完成，總耗時 {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'最佳驗證準確率: {best_acc:.4f}')
    
    # 載入最佳模型權重
    model.load_state_dict(best_model_wts)
    return model, history

def save_training_plots(history, save_dir):
    """
    保存訓練過程的損失和準確率曲線
    
    Args:
        history (dict): 包含訓練和驗證損失及準確率的字典
        save_dir (str): 保存圖表的目錄
    """
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'))
    plt.close()
    
    print(f"訓練歷史圖表已保存至 {os.path.join(save_dir, 'training_history.png')}")