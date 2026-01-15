# VRCI Platform Installation Guide

Complete step-by-step instructions for setting up the VRCI experimental platform on your local machine or server.

**Contact**: admin@gy4k.com  
**Last Updated**: January 15, 2026  
**Platform Version**: 1.0.0

---

## ⚠️ Important Notice About Training Data

**Data Confidentiality Statement**

The actual training data used in our research cannot be publicly released due to:
1. Proprietary experimental design parameters from our laboratory's simulation environment
2. Commercial confidentiality agreements with partner companies

**What IS Provided:**
- ✅ Complete model architectures and training procedures
- ✅ Data generation scripts based on public mathematical models
- ✅ All configuration parameters from industry standards
- ✅ Reproducible training methodology

**What to Expect:**
- Models trained on generated data will show similar trends and order-of-magnitude performance
- Exact numerical values may vary slightly (±5-10%) from paper results
- This approach balances scientific reproducibility with legitimate confidentiality requirements

For detailed data generation methodology, see `backend/training/generate_training_data.py`.

---

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation Steps](#installation-steps)
3. [Verification](#verification)
4. [Troubleshooting](#troubleshooting)
5. [Advanced Configuration](#advanced-configuration)

---

## System Requirements

### Minimum Requirements

| Component | Specification |
|-----------|--------------|
| **OS** | Ubuntu 20.04+, macOS 12+, Windows 10+ (WSL2) |
| **Python** | 3.11 or higher |
| **CPU** | Intel Core i5-10400 or AMD Ryzen 5 3600 (6 cores) |
| **GPU** | NVIDIA GTX 1660 Ti (6GB VRAM) with CUDA 12.1+ |
| **RAM** | 16GB DDR4 |
| **Storage** | 10GB free space (SSD recommended) |
| **Network** | Broadband internet for initial setup |

**Expected Performance**:
- Prediction latency: 2-5 seconds
- Training time: 15-20 minutes per model
- Dashboard refresh: 1-2 seconds

### Recommended Requirements (Used in Paper)

| Component | Specification |
|-----------|--------------|
| **OS** | Ubuntu 22.04 LTS or macOS 14+ |
| **Python** | 3.11.7 |
| **CPU** | Intel Core i9-14900K (24 cores, 6.0 GHz boost) |
| **GPU** | NVIDIA GeForce RTX 4090 (24GB VRAM, 16,384 CUDA cores) |
| **RAM** | 64GB DDR5-5600 |
| **Storage** | 2TB NVMe PCIe 4.0 SSD |

**Expected Performance**:
- Prediction latency: <100ms
- Training time: 1.5-3.5 hours for all 5 models
- Dashboard refresh: Real-time (<50ms)

---

## Installation Steps

### Step 1: Clone Repository

```bash
git clone https://github.com/PolyUDavid/VRCI_Commander_Center.git
cd vrci-platform
```

### Step 2: Create Virtual Environment

**Linux/macOS**:
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (PowerShell)**:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Verify activation**:
```bash
which python  # Should show path inside venv/
python --version  # Should be 3.11+
```

### Step 3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA support (for NVIDIA GPU)
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements.txt
```

**For CPU-only installation** (not recommended):
```bash
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### Step 4: Verify CUDA Installation

```bash
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

**Expected output** (with GPU):
```
CUDA Available: True
Device: NVIDIA GeForce RTX 4090
```

### Step 5: Download Model Checkpoints

The trained model checkpoints are included in the repository under `backend/models/`. Verify they exist:

```bash
ls -lh backend/models/
```

**Expected files** (~200MB total):
```
latency_lstm_enhanced.pth      (67MB)
energy_rwkv_enhanced.pth       (29MB)
coverage_mamba3.pth            (49MB)
consensus_retnet.pth           (37MB)
carbon_lightts.pth             (19MB)
```

If missing, contact admin@gy4k.com for access.

### Step 6: Generate Initial Dataset

```bash
cd backend
python generate_paper_dataset.py
```

**Expected output**:
```
Generating 2000 samples...
  Generating 800 urban scenarios...
  Generating 600 highway scenarios...
  Generating 400 intersection scenarios...
  Generating 200 rural scenarios...

✓ Saved JSON: ../data/vrci_paper_dataset.json
✓ Saved CSV: ../data/vrci_paper_dataset.csv

============================================================
DATASET STATISTICS
============================================================
Total Samples: 2000
Latency Reduction: Mean: 67.2% ± 3.1%
Energy Savings: Mean: 42.5% ± 2.8%
...
```

### Step 7: Start Backend API Server

```bash
cd backend
python api_server_ai.py
```

**Expected output**:
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8001 (Press CTRL+C to quit)
```

**Keep this terminal open!**

### Step 8: Start Frontend Server

Open a **new terminal**, activate venv, and run:

```bash
cd frontend
python -m http.server 8080
```

**Expected output**:
```
Serving HTTP on 0.0.0.0 port 8080 (http://0.0.0.0:8080/) ...
```

**Keep this terminal open!**

### Step 9: Access Dashboard

Open your web browser and navigate to:

```
http://localhost:8080/dashboard_ultimate.html
```

You should see the **Command Center** page with:
- Real-time KPI cards
- 3D Beijing map
- Network status charts
- AI model status

---

## Verification

### Test 1: API Health Check

```bash
curl http://localhost:8001/health
```

**Expected response**:
```json
{
  "status": "healthy",
  "models_loaded": 5,
  "version": "1.0.0"
}
```

### Test 2: Latency Prediction

```bash
curl -X POST http://localhost:8001/api/predict/latency \
  -H "Content-Type: application/json" \
  -d '{
    "vehicle_density": 80.0,
    "data_size_mb": 2.0,
    "weather": "clear",
    "time_of_day": "morning",
    "backhaul_latency_ms": 80.0
  }'
```

**Expected response**:
```json
{
  "status": "success",
  "ccc_latency_ms": 145.3,
  "dec_latency_ms": 47.8,
  "latency_reduction_percent": 67.1,
  "confidence_score": 0.94
}
```

### Test 3: Dashboard Functionality

1. Navigate to **Simulation Dashboard** (left sidebar)
2. Adjust "Vehicle Density" slider to 100 veh/km
3. Click "Run Simulation"
4. Verify:
   - Progress bar appears
   - KPI cards update
   - Charts refresh
   - No console errors (F12 → Console)

---

## Troubleshooting

### Issue 1: CUDA Not Available

**Symptom**: `CUDA Available: False`

**Solutions**:
1. **Check NVIDIA driver**:
   ```bash
   nvidia-smi
   ```
   Should show GPU info. If not, install latest driver from [NVIDIA website](https://www.nvidia.com/Download/index.aspx).

2. **Reinstall PyTorch with CUDA**:
   ```bash
   pip uninstall torch torchvision torchaudio
   pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
   ```

3. **Verify CUDA toolkit**:
   ```bash
   nvcc --version
   ```
   Should show CUDA 12.1+. If not, install from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads).

### Issue 2: Port Already in Use

**Symptom**: `OSError: [Errno 48] Address already in use`

**Solutions**:
1. **Find process using port 8001**:
   ```bash
   lsof -i :8001
   ```
   
2. **Kill process**:
   ```bash
   kill -9 <PID>
   ```
   
3. **Or use different port**:
   ```bash
   uvicorn api_server_ai:app --host 0.0.0.0 --port 8002
   ```

### Issue 3: Model Files Missing

**Symptom**: `FileNotFoundError: [Errno 2] No such file or directory: 'models/latency_lstm_enhanced.pth'`

**Solutions**:
1. **Check if files exist**:
   ```bash
   ls backend/models/
   ```

2. **If missing, contact for access**:
   Email: admin@gy4k.com
   Subject: "VRCI Model Checkpoints Request"

3. **Or retrain models** (requires 2-4 hours):
   ```bash
   cd backend
   python train_all_models.py
   ```

### Issue 4: Dashboard Not Loading

**Symptom**: Blank page or "Connection refused"

**Solutions**:
1. **Check both servers are running**:
   - Backend: `curl http://localhost:8001/health`
   - Frontend: `curl http://localhost:8080`

2. **Check browser console** (F12 → Console):
   - Look for CORS errors
   - Look for 404 errors

3. **Clear browser cache**:
   - Chrome: Ctrl+Shift+Delete
   - Firefox: Ctrl+Shift+Delete
   - Safari: Cmd+Option+E

4. **Try different browser**:
   - Recommended: Chrome 120+, Firefox 120+, Safari 17+

### Issue 5: Slow Predictions (>5 seconds)

**Symptom**: Dashboard takes long time to update

**Solutions**:
1. **Check GPU utilization**:
   ```bash
   nvidia-smi
   ```
   Should show GPU memory usage ~2-4GB.

2. **Reduce batch size** (in `api_server_ai.py`):
   ```python
   BATCH_SIZE = 32  # Reduce from 64
   ```

3. **Enable mixed precision** (in model inference):
   ```python
   with torch.cuda.amp.autocast():
       predictions = model(inputs)
   ```

### Issue 6: Out of Memory (OOM)

**Symptom**: `RuntimeError: CUDA out of memory`

**Solutions**:
1. **Reduce model batch size**:
   ```python
   BATCH_SIZE = 16  # In api_server_ai.py
   ```

2. **Clear GPU cache**:
   ```python
   torch.cuda.empty_cache()
   ```

3. **Use CPU inference** (slower):
   ```python
   device = torch.device('cpu')
   ```

---

## Advanced Configuration

### Custom Port Configuration

Edit `backend/api_server_ai.py`:

```python
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)  # Change port
```

Edit `frontend/dashboard_ultimate.html`:

```javascript
const API_BASE_URL = 'http://localhost:8002';  // Match backend port
```

### Enable HTTPS (Production)

```bash
# Generate self-signed certificate
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

# Run with SSL
uvicorn api_server_ai:app --host 0.0.0.0 --port 8001 --ssl-keyfile key.pem --ssl-certfile cert.pem
```

### Multi-GPU Support

Edit `backend/api_server_ai.py`:

```python
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
```

### Docker Deployment

```dockerfile
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3.11 python3-pip
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY backend/ /app/backend/
COPY frontend/ /app/frontend/
COPY data/ /app/data/

WORKDIR /app
CMD ["uvicorn", "backend.api_server_ai:app", "--host", "0.0.0.0", "--port", "8001"]
```

Build and run:
```bash
docker build -t vrci-platform .
docker run --gpus all -p 8001:8001 -p 8080:8080 vrci-platform
```

---

## Next Steps

After successful installation:

1. **Explore Dashboard**: Navigate through all pages (Command Center, Latency, Energy, etc.)
2. **Run Simulations**: Adjust parameters and observe metric changes
3. **Generate Data**: Export CSV/JSON for analysis
4. **Validate Results**: Compare with paper metrics (see `docs/REPRODUCIBILITY.md`)
5. **Customize**: Modify `config/config_standard.yaml` for your use case

---

## Support

If you encounter issues not covered here:

1. **Check GitHub Issues**: https://github.com/PolyUDavid/VRCI_Commander_Center/issues
2. **Email Support**: admin@gy4k.com
3. **Documentation**: See `docs/` folder for more guides

---

**Installation Time**: 15-30 minutes  
**Difficulty**: Intermediate  
**Prerequisites**: Basic Python knowledge, command line familiarity

**Last Updated**: January 15, 2026  
**Platform Version**: 1.0.0
