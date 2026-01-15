# ⚠️ DATA NOTICE: Training data generated from public models due to confidentiality. See generate_training_data.py
"""Consensus增强模型训练"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import os

from .model_enhanced import ConsensusRetNet_Enhanced

class ConsensusDataset(Dataset):
    def __init__(self, csv_path, normalize=True):
        self.df = pd.read_csv(csv_path)
        self.normalize = normalize
        
        # Consensus特定的列
        self.feature_cols = ['tps_required', 'latency_max_ms', 'energy_budget_w']
        
        self.X = self.df[self.feature_cols].values.astype(np.float32)
        
        # 将optimal_mechanism转换为one-hot编码
        mechanisms = ['PoW', 'PoS', 'PBFT', 'DPoS', 'PoL']
        self.y = np.zeros((len(self.df), 5), dtype=np.float32)
        for i, mech in enumerate(self.df['optimal_mechanism']):
            if mech in mechanisms:
                self.y[i, mechanisms.index(mech)] = 1
        
        if normalize:
            self.X_scaler = StandardScaler()
            self.X = self.X_scaler.fit_transform(self.X)
        else:
            self.X_scaler = None
        
        print(f"✅ 加载Consensus数据: {len(self)} 样本")
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)

def train_model(csv_path, epochs=100, batch_size=128, lr=0.0003):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n{'='*80}\n🚀 Consensus增强模型训练\n{'='*80}\n设备: {device}, Epochs: {epochs}\n")
    
    dataset = ConsensusDataset(csv_path)
    train_size = int(0.85 * len(dataset))
    train_ds, val_ds = random_split(dataset, [train_size, len(dataset) - train_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    model = ConsensusRetNet_Enhanced(input_dim=3, hidden_dim=128, num_classes=5).to(device)
    print(f"🤖 参数量: {sum(p.numel() for p in model.parameters()):,}\n")
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    
    best_acc, best_epoch, patience, patience_counter = 0, 0, 25, 0
    
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for batch_X, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
        
        model.eval()
        val_losses, all_preds, all_targets = [], [], []
        with torch.no_grad():
            for batch_X, batch_y in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False):
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                val_losses.append(criterion(outputs, batch_y).item())
                all_preds.append(torch.sigmoid(outputs).cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())
        
        all_preds = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)
        
        # 转换为类别
        pred_classes = all_preds.argmax(axis=1)
        true_classes = all_targets.argmax(axis=1)
        current_acc = accuracy_score(true_classes, pred_classes)
        
        print(f"\n📊 Epoch [{epoch+1}/{epochs}]")
        print(f"   Accuracy: {current_acc:.6f}")
        
        if current_acc > best_acc:
            best_acc, best_epoch, patience_counter = current_acc, epoch+1, 0
            os.makedirs("ai_models/consensus/checkpoints", exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'accuracy': best_acc,
                'epoch': best_epoch
            }, "ai_models/consensus/checkpoints/consensus_enhanced_best.pth")
            print(f"💾 New best! Acc={best_acc:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n🛑 Early stop at epoch {epoch+1}")
                break
        scheduler.step()
    
    print(f"\n🎉 训练完成! 最佳Accuracy: {best_acc:.6f} (Epoch {best_epoch})\n")
    return model

if __name__ == "__main__":
    BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    train_model(os.path.join(BASE, "data/training/consensus/consensus_train_8k.csv"))
