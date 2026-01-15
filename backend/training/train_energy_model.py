# ⚠️ DATA NOTICE: Training data generated from public models due to confidentiality. See generate_training_data.py
"""Energy增强模型训练"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from tqdm import tqdm
import os
import time

from .model_enhanced import EnergyRWKV_Enhanced

class EnergyDataset(Dataset):
    def __init__(self, csv_path, normalize=True):
        self.df = pd.read_csv(csv_path)
        self.normalize = normalize
        
        # Energy特定的列
        self.feature_cols = ['density_veh_per_km', 'data_size_mb', 'tx_power_cloud_w', 
                            'tx_power_edge_w', 'pue']
        self.target_cols = ['ccc_total_mj', 'dec_total_mj']
        
        self.X = self.df[self.feature_cols].values.astype(np.float32)
        self.y = self.df[self.target_cols].values.astype(np.float32)
        
        if normalize:
            self.X_scaler = StandardScaler()
            self.y_scaler = StandardScaler()
            self.X = self.X_scaler.fit_transform(self.X)
            self.y = self.y_scaler.fit_transform(self.y)
        else:
            self.X_scaler = self.y_scaler = None
        
        print(f"✅ 加载Energy数据: {len(self)} 样本")
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)

def train_model(csv_path, epochs=150, batch_size=128, lr=0.0003):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n{'='*80}\n🚀 Energy增强模型训练\n{'='*80}\n设备: {device}, Epochs: {epochs}\n")
    
    dataset = EnergyDataset(csv_path)
    train_size = int(0.85 * len(dataset))
    train_ds, val_ds = random_split(dataset, [train_size, len(dataset) - train_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    model = EnergyRWKV_Enhanced(input_dim=5, hidden_dim=128, output_dim=2).to(device)
    print(f"🤖 参数量: {sum(p.numel() for p in model.parameters()):,}\n")
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2)
    
    best_r2, best_epoch, patience, patience_counter = -float('inf'), 0, 35, 0
    
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
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())
        
        all_preds = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)
        
        if dataset.y_scaler:
            all_preds = dataset.y_scaler.inverse_transform(all_preds)
            all_targets = dataset.y_scaler.inverse_transform(all_targets)
        
        current_r2 = r2_score(all_targets, all_preds)
        mae = mean_absolute_error(all_targets, all_preds)
        r2_ccc = r2_score(all_targets[:, 0], all_preds[:, 0])
        r2_dec = r2_score(all_targets[:, 1], all_preds[:, 1])
        
        print(f"\n📊 Epoch [{epoch+1}/{epochs}]")
        print(f"   Overall R²: {current_r2:.6f}, MAE: {mae:.4f}")
        print(f"   R²_CCC: {r2_ccc:.6f}, R²_DEC: {r2_dec:.6f}")
        
        if current_r2 > best_r2:
            best_r2, best_epoch, patience_counter = current_r2, epoch+1, 0
            os.makedirs("ai_models/energy/checkpoints", exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'r2': best_r2,
                'epoch': best_epoch
            }, "ai_models/energy/checkpoints/energy_enhanced_best.pth")
            print(f"💾 New best! R²={best_r2:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n🛑 Early stop at epoch {epoch+1}")
                break
        scheduler.step()
    
    print(f"\n🎉 训练完成! 最佳R²: {best_r2:.6f} (Epoch {best_epoch})\n")
    return model

if __name__ == "__main__":
    BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    train_model(os.path.join(BASE, "data/training/energy/energy_train_12k.csv"))
