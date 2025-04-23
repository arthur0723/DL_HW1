# Mini-ImageNet Classification - Task B

本專案實現了使用少量有效層(2-4層)的神經網絡進行圖像分類任務，目標是達到ResNet34至少90%的性能。

## 專案結構

```
Task_B/
├── models/
│   ├── baseline.py         # ResNet34 基準模型
│   ├── four_layer_net.py   # FourLayerNet 實現
│   └── four_layer_net_ablation.py # FourLayerNet 消融實驗模型
├── utils/
│   ├── dataset.py          # 資料集類和轉換
│   ├── train.py            # 訓練函數
│   └── evaluate.py         # 評估函數
├── experiments/
│   ├── baseline/           # 基準模型實驗結果
│   ├── four_layer/         # FourLayerNet實驗結果
│   └── four_layer_ablation/ # 消融實驗結果
├── README.md               # 專案說明
└── main.py                 # 主執行檔
```

## 重現實驗步驟

### 1. 訓練基準模型

```bash
# 訓練ResNet34基準模型
python main.py --model resnet34 --batch_size 64 --epochs 10 --lr 0.01
```

### 2. 訓練四層網絡模型

```bash
# 訓練FourLayerNet模型
python main.py --model fourlayernet --batch_size 64 --epochs 50 --lr 0.001
```

### 3. 評估模型

```bash
# 評估基準模型
python main.py --model resnet34 --mode eval --checkpoint experiments/baseline/best_model.pth

# 評估四層網絡模型
python main.py --model fourlayernet --mode eval --checkpoint experiments/four_layer/best_model.pth
```

### 4. 消融實驗

消融實驗旨在評估注意力機制對FourLayerNet性能的貢獻。我們實現了一個替代模型，將原始模型中的注意力機制替換為普通卷積層。

#### 4.1 訓練消融模型

```bash
# 訓練不含注意力機制的FourLayerNet變體
python main.py --model fourlayerablation --batch_size 64 --epochs 20 --lr 0.001
```

#### 4.2 評估消融模型

```bash
# 評估消融實驗模型
python main.py --model fourlayerablation --mode eval --checkpoint experiments/four_layer_ablation/best_model.pth
```

消融實驗結果將保存在`experiments/four_layer_ablation/`目錄中，可與原始模型結果進行比較分析。