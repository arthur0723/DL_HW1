import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

from models.baseline import create_resnet34
from models.four_layer_net import FourLayerNet
from models.four_layer_net_ablation import FourLayerAblation
from utils.dataset import load_data
from utils.train import train_model, save_training_plots
from utils.evaluate import evaluate_model, calculate_metrics, calculate_params_and_flops, save_evaluation_results

def set_seed(seed=42):
    """
    設置隨機種子以獲得可重現的結果
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    # 解析命令行參數
    parser = argparse.ArgumentParser(description='Mini-ImageNet Classification')
    parser.add_argument('--model', type=str, required=True, 
                       choices=['resnet34', 'fourlayernet', 'fourlayerablation'],
                       help='Model architecture to use')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'],
                       help='Mode: train or evaluate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint for evaluation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--data_dir', type=str, default='', help='Directory for image data')
    
    args = parser.parse_args()
    
    # 設置隨機種子
    set_seed(args.seed)
    
    # 設定裝置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用設備: {device}")
    
    # 設定路徑
    data_dir = args.data_dir
    train_txt = 'train.txt'
    val_txt = 'val.txt'
    test_txt = 'test.txt'
    
    # 創建實驗目錄
    if args.model == 'resnet34':
        model_save_dir = 'experiments/baseline'
    elif args.model == 'fourlayernet':
        model_save_dir = 'experiments/four_layer'
    elif args.model == 'fourlayerablation':
        model_save_dir = 'experiments/four_layer_ablation'
    
    os.makedirs(model_save_dir, exist_ok=True)
    
    # 載入數據
    print("載入數據...")
    image_datasets, dataloaders, dataset_sizes, class_names = load_data(
        data_dir, train_txt, val_txt, test_txt, args.batch_size
    )
    
    # 創建模型
    print(f"創建 {args.model} 模型...")
    if args.model == 'resnet34':
        model = create_resnet34(class_names)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    elif args.model == 'fourlayernet':
        model = FourLayerNet(num_classes=class_names)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6
        )
    elif args.model == 'fourlayerablation':
        model = FourLayerAblation(num_classes=class_names)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6
        )
    
    model = model.to(device)
    
    # 定義損失函數
    criterion = nn.CrossEntropyLoss()
    
    if args.mode == 'train':
        print(f"開始訓練 {args.model} 模型，共 {args.epochs} 個周期...")
        
        # 訓練模型
        model, history = train_model(
            model, dataloaders, dataset_sizes, criterion, optimizer, scheduler,
            num_epochs=args.epochs, device=device
        )
        
        # 保存模型
        torch.save(model.state_dict(), os.path.join(model_save_dir, 'best_model.pth'))
        print(f"模型已保存至 {os.path.join(model_save_dir, 'best_model.pth')}")
        
        # 保存訓練圖表
        save_training_plots(history, model_save_dir)
        
        # 計算模型參數和計算量
        params, flops = calculate_params_and_flops(model)
        
        # 評估模型
        print("評估訓練後的模型...")
        test_acc, test_preds, test_labels = evaluate_model(
            model, dataloaders['test'], device, '測試集'
        )
        
        # 計算所有性能指標
        metrics = calculate_metrics(test_preds, test_labels)
        
        # 整合結果
        results = {
            'model': args.model,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'test_accuracy': test_acc,
            'params': params,
            'flops': flops
        }
        
        # 保存評估結果
        save_evaluation_results(results, metrics, model_save_dir)
        
    else:  # Evaluation mode
        print(f"載入檢查點 {args.checkpoint} 並評估模型...")
        
        # 載入模型
        model.load_state_dict(torch.load(args.checkpoint))
        model.eval()
        
        # 評估模型
        train_acc, train_preds, train_labels = evaluate_model(
            model, dataloaders['train'], device, '訓練集'
        )
        val_acc, val_preds, val_labels = evaluate_model(
            model, dataloaders['val'], device, '驗證集'
        )
        test_acc, test_preds, test_labels = evaluate_model(
            model, dataloaders['test'], device, '測試集'
        )
        
        # 計算模型參數和計算量
        params, flops = calculate_params_and_flops(model)
        
        # 計算測試集的性能指標
        metrics = calculate_metrics(test_preds, test_labels)
        
        # 整合結果
        results = {
            'model': args.model,
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'test_accuracy': test_acc,
            'params': params,
            'flops': flops
        }
        
        # 保存評估結果
        save_evaluation_results(results, metrics, model_save_dir)
        
        print("評估完成!")
        print(f"測試集準確率: {test_acc:.4f}")
        print(f"模型參數數量: {params:,}")
        if flops is not None:
            print(f"模型計算複雜度 (FLOPs): {flops:,}")

if __name__ == "__main__":
    main()