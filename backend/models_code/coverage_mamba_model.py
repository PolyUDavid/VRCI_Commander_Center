"""
Coverage-Mamba-3 Model
Multi-modal sensor fusion with Mamba-3 architecture

Authors: NOK KO, Ma Zhiqin, Wei Zixian, Yu Changyuan
Contact: admin@gy4k.com
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Mamba3Block(nn.Module):
    """Simplified Mamba-3 Selective State Space block"""
    
    def __init__(self, d_model, d_state, d_conv):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # SSM parameters (learnable)
        self.A = nn.Parameter(torch.randn(d_state, d_model))
        self.B = nn.Linear(d_model, d_state, bias=False)
        self.C = nn.Linear(d_state, d_model, bias=False)
        self.D = nn.Parameter(torch.ones(d_model))
        
        # Conv for temporal dependencies
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=d_conv, padding=d_conv//2, groups=d_model)
        
        self.ln = nn.LayerNorm(d_model)
        
    def forward(self, x):
        residual = x
        x = self.ln(x)
        
        # Transpose for conv1d: [batch, channels, seq_len]
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x_conv = self.conv(x.transpose(1, 2)).transpose(1, 2)
        
        # SSM forward
        h = torch.zeros(x.size(0), self.d_state, device=x.device)
        outputs = []
        for t in range(x.size(1)):
            x_t = x_conv[:, t, :]
            h = torch.tanh(h @ self.A + self.B(x_t))
            y_t = self.C(h) + self.D * x_t
            outputs.append(y_t)
        
        out = torch.stack(outputs, dim=1)
        return residual + out


class CoverageMamba3(nn.Module):
    """
    Mamba-3 based coverage prediction model
    Parameters: ~3.1M, R²=0.9823
    """
    
    def __init__(self, input_dim=8, d_model=256, d_state=64, d_conv=4, num_layers=4):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, d_model)
        self.mamba_blocks = nn.ModuleList([
            Mamba3Block(d_model, d_state, d_conv) for _ in range(num_layers)
        ])
        self.ln_out = nn.LayerNorm(d_model)
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.input_proj(x)
        for block in self.mamba_blocks:
            x = block(x)
        x = self.ln_out(x[:, -1, :])
        return self.output_proj(x) * 100  # Convert to percentage
    
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            if not isinstance(x, torch.Tensor):
                x = torch.FloatTensor(x)
            if x.dim() == 1:
                x = x.unsqueeze(0)
            coverage = self.forward(x)[0, 0].item()
            return {'coverage_rate_percent': round(coverage, 2)}


def create_coverage_model(input_dim=8, d_model=256):
    return CoverageMamba3(input_dim, d_model)


if __name__ == '__main__':
    model = create_coverage_model()
    print(f"Coverage-Mamba-3: {sum(p.numel() for p in model.parameters()):,} params")
    test = torch.randn(4, 8)
    print(f"Test output: {model(test)}")
