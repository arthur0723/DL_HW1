import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class MiniImageNetDataset(Dataset):
    """
    Mini-ImageNet 資料集
    """
    def __init__(self, txt_path, img_dir, transform=None):
        """
        初始化資料集
        
        Args:
            txt_path (str): 包含圖像路徑和標籤的文本文件路徑
            img_dir (str): 圖像目錄的路徑
            transform (callable, optional): 應用於圖像的轉換
        """
        self.img_dir = img_dir
        self.transform = transform
        self.img_labels = []
        
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    img_path = parts[0]
                    label = int(parts[1])
                    self.img_labels.append((img_path, label))
    
    def __len__(self):
        """返回資料集的大小"""
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        """獲取指定索引的樣本"""
        img_path, label = self.img_labels[idx]
        image = Image.open(os.path.join(self.img_dir, img_path)).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_transforms():
    """
    獲取數據轉換
    
    Returns:
        dict: 包含訓練、驗證和測試轉換的字典
    """
    return {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

def load_data(data_dir, train_txt, val_txt, test_txt, batch_size=64, num_workers=4):
    """
    加載資料集並創建資料加載器
    
    Args:
        data_dir (str): 圖像目錄路徑
        train_txt (str): 訓練集文本文件路徑
        val_txt (str): 驗證集文本文件路徑
        test_txt (str): 測試集文本文件路徑
        batch_size (int, optional): 批次大小
        num_workers (int, optional): 數據加載的工作進程數
    
    Returns:
        tuple: (image_datasets, dataloaders, dataset_sizes, class_names)
    """
    # 獲取數據轉換
    data_transforms = get_transforms()
    
    # 創建數據集
    image_datasets = {
        'train': MiniImageNetDataset(train_txt, data_dir, data_transforms['train']),
        'val': MiniImageNetDataset(val_txt, data_dir, data_transforms['val']),
        'test': MiniImageNetDataset(test_txt, data_dir, data_transforms['test'])
    }
    
    # 創建數據加載器
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=num_workers),
        'val': DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False, num_workers=num_workers),
        'test': DataLoader(image_datasets['test'], batch_size=batch_size, shuffle=False, num_workers=num_workers)
    }
    
    # 計算數據集大小
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    
    # 獲取類別數量
    class_names = len(set([label for _, label in image_datasets['train'].img_labels]))
    
    print(f"訓練集大小: {dataset_sizes['train']}")
    print(f"驗證集大小: {dataset_sizes['val']}")
    print(f"測試集大小: {dataset_sizes['test']}")
    print(f"類別數量: {class_names}")
    
    return image_datasets, dataloaders, dataset_sizes, class_names