"""
Latency-LSTM Enhanced Model Training Script
训练目标：R² > 0.95

⚠️ IMPORTANT DATA NOTICE / 重要数据说明:
===========================================
Training data used in this script is generated from publicly available 
mathematical models and industry standards, as the proprietary experimental 
data from our laboratory cannot be released due to:
1. Specific parameter fine-tuning in simulation design
2. Commercial confidentiality agreements with partner companies

The training procedure and model architecture are fully reproducible.
Performance may vary slightly (±5-10%) from paper results due to 
different training data distributions.

本脚本使用的训练数据基于公开模型生成，因实验室专有参数和
商业保密协议，无法公开原始数据。模型架构和训练流程完全可复现。

Contact: admin@gy4k.com
===========================================
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tqdm import tqdm
import os
import time

from .model_enhanced import LatencyLSTM_Enhanced

class LatencyDataset(Dataset):
    def __init__(self, csv_path, normalize=True):
        self.df = pd.read_csv(csv_path)
        self.normalize = normalize
        
        # 使用所有可用特征
        self.feature_cols = ['density_veh_per_km', 'data_size_mb', 'backhaul_latency_ms',
                            'bandwidth_cloud_mbps', 'bandwidth_edge_gbps', 
                            'processing_power_cloud_ghz', 'processing_power_edge_ghz',
                            'queue_service_rate', 'congestion_factor']
        self.target_cols = ['ccc_latency_ms', 'dec_latency_ms', 'latency_reduction_percent']
        
        self.X = self.df[self.feature_cols].values.astype(np.float32)
        self.y = self.df[self.target_cols].values.astype(np.float32)
        
        if normalize:
            self.X_scaler = StandardScaler()
            self.y_scaler = StandardScaler()
            self.X = self.X_scaler.fit_transform(self.X)
            self.y = self.y_scaler.fit_transform(self.y)
        else:
            self.X_scaler = self.y_scaler = None
        
        print(f"✅ 加载Latency数据: {len(self)} 样本, 特征:{len(self.feature_cols)}, 目标:{len(self.target_cols)}")
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])
    
    def denormalize(self, y):
        if self.y_scaler is not None:
            return self.y_scaler.inverse_transform(y)
        return y


def train_model(csv_path, epochs=250, batch_size=128, lr=0.0003, patience=35):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n{'='*80}")
    print(f"🚀 开始训练: Latency-LSTM-Enhanced")
    print(f"{'='*80}")
    print(f"📊 训练配置:")
    print(f"   设备: {device}")
    print(f"   总Epochs: {epochs}")
    print(f"   Batch Size: {batch_size}")
    print(f"   学习率: {lr}")
    print(f"   Early Stop Patience: {patience}")
    print(f"{'='*80}\n")
    
    # 加载数据
    dataset = LatencyDataset(csv_path)
    train_size = int(0.85 * len(dataset))  # 85% train, 15% val
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"📊 数据划分: Train={train_size}, Val={val_size}\n")
    
    # 创建增强模型
    model = LatencyLSTM_Enhanced(
        input_dim=9,
        hidden_dim=128,
        num_layers=3,
        output_dim=3,
        dropout_rate=0.2
    ).to(device)
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"🤖 模型参数量: {param_count:,} ({param_count/1e6:.2f}M)\n")
    
    # 优化器和调度器
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # Cosine Annealing with Warmup
    warmup_epochs = 20
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=30, T_mult=2, eta_min=lr/100
    )
    
    # Early Stopping
    best_val_r2 = -float('inf')
    best_epoch = 0
    patience_counter = 0
    
    # 训练历史
    history = {'train_loss': [], 'val_loss': [], 'val_r2': [], 'lr': []}
    
    print(f"🏋️  开始训练...\n")
    start_time = time.time()
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_losses = []
        epoch_start = time.time()
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        for batch_X, batch_y in train_bar:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_losses.append(loss.item())
            train_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = np.mean(train_losses)
        
        # 验证阶段
        model.eval()
        val_losses = []
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False)
            for batch_X, batch_y in val_bar:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                val_losses.append(loss.item())
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())
        
        avg_val_loss = np.mean(val_losses)
        
        # 反归一化并计算R²
        all_preds = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)
        
        all_preds_orig = dataset.denormalize(all_preds)
        all_targets_orig = dataset.denormalize(all_targets)
        
        val_r2 = r2_score(all_targets_orig, all_preds_orig)
        val_mae = mean_absolute_error(all_targets_orig, all_preds_orig)
        
        # 计算每个输出的R²
        r2_ccc = r2_score(all_targets_orig[:, 0], all_preds_orig[:, 0])
        r2_dec = r2_score(all_targets_orig[:, 1], all_preds_orig[:, 1])
        r2_reduction = r2_score(all_targets_orig[:, 2], all_preds_orig[:, 2])
        
        # 计算平均延迟降低
        avg_reduction_true = all_targets_orig[:, 2].mean()
        avg_reduction_pred = all_preds_orig[:, 2].mean()
        
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录历史
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_r2'].append(val_r2)
        history['lr'].append(current_lr)
        
        # 打印进度
        print(f"\n{'='*80}")
        print(f"📊 Epoch [{epoch+1}/{epochs}] - {epoch/(epochs)*100:.1f}% Complete")
        print(f"{'='*80}")
        print(f"🏋️  Training:")
        print(f"   Loss: {avg_train_loss:.6f}")
        print(f"\n✅ Validation:")
        print(f"   Loss: {avg_val_loss:.6f}")
        print(f"   Overall R²: {val_r2:.6f}")
        print(f"   MAE: {val_mae:.4f}")
        print(f"   R²_CCC: {r2_ccc:.6f}")
        print(f"   R²_DEC: {r2_dec:.6f}")
        print(f"   R²_Reduction: {r2_reduction:.6f}")
        print(f"\n📈 Latency Reduction:")
        print(f"   True Avg: {avg_reduction_true:.2f}%")
        print(f"   Pred Avg: {avg_reduction_pred:.2f}%")
        print(f"   Error: {abs(avg_reduction_true - avg_reduction_pred):.2f}%")
        print(f"\n🔧 Optimization:")
        print(f"   Learning Rate: {current_lr:.8f}")
        print(f"   Epoch Time: {epoch_time:.2f}s")
        print(f"={'='*80}\n")
        
        # 保存最佳模型
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_epoch = epoch + 1
            patience_counter = 0
            
            os.makedirs('ai_models/latency/checkpoints', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_r2': val_r2,
                'val_loss': avg_val_loss,
                'X_scaler': dataset.X_scaler,
                'y_scaler': dataset.y_scaler
            }, 'ai_models/latency/checkpoints/latency_enhanced_best.pth')
            
            print(f"💾 New best model saved! R²={val_r2:.6f}\n")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n🛑 Early stopping triggered at epoch {epoch+1}")
                print(f"   Best R² was {best_val_r2:.6f} at epoch {best_epoch}\n")
                break
        
        scheduler.step()
    
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"🎉 训练完成!")
    print(f"{'='*80}")
    print(f"⏱️  总用时: {total_time/60:.2f}分钟")
    print(f"🏆 最佳R²: {best_val_r2:.6f} (Epoch {best_epoch})")
    print(f"{'='*80}\n")
    
    return model, history


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    CSV_PATH = os.path.join(BASE_DIR, "data/training/latency/latency_train_15k.csv")
    
    model, history = train_model(CSV_PATH)
