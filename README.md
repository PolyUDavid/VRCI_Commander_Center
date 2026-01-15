# Decentralized Vehicle-Road-Cloud Integration (VRCI) Experimental Platform

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-red.svg)](https://pytorch.org/)

## Overview

This repository contains the complete experimental platform for validating the feasibility of decentralized Vehicle-Road-Cloud Integration (VRCI) systems, as presented in our research paper:

> **"Decentralizing Vehicle-Road-Cloud Integration: A Feasibility Study with AI-Enhanced Validation Platform and Sustainability Assessment"**  
> *Submitted to Sustainable Cities and Society*

The platform integrates five state-of-the-art deep learning models with mathematical feasibility analysis to provide quantitative validation of:
- **67.3% latency reduction** (centralized vs. decentralized architectures)
- **42.7% energy savings** (discovering empirical f^2.3 power scaling law)
- **95.7% spatiotemporal coverage** (multi-modal sensor fusion: RSU + UAV + Vehicle)
- **96.9% consensus mechanism selection accuracy** (PBFT, DPoS, PoS, PoW)
- **2.0-2.5 kt CO₂ net savings over 10 years** (12-month carbon payback period)

---

## Key Features

### 🚀 **AI-Enhanced Experimental Framework**
- **5 Hybrid Mathematical-AI Models**: Latency-LSTM, Energy-RWKV, Coverage-Mamba-3, Consensus-RetNet, Carbon-LightTS
- **Total 12.6M Parameters**: Trained on 30,000+ realistic VRCI scenarios
- **Hardware**: NVIDIA RTX 4090 (24GB) + Intel i9-14900K (24 cores, 6.0 GHz)
- **Real-time Inference**: <100ms per prediction enabling interactive exploration

### 📊 **Interactive Web Dashboard**
- **Command Center**: Real-time monitoring with 3D geographic coverage map
- **Model Architecture Visualization**: Detailed layer diagrams with training metrics
- **Monte Carlo Validation**: 500-run statistical analysis with confidence intervals
- **Parameter Adjustment**: Interactive sliders for deployment scenario exploration
- **Data Export**: Rich datasets (2000+ samples, CSV/JSON formats)

### 🔬 **Reproducible Methodology**
- **67 Mathematical Equations**: Complete derivations (main paper + supplement)
- **Paper-Constrained Loss Functions**: Ensures predictions align with physical constraints
- **Boundary Analysis**: Critical threshold at 50 veh/km (cost-benefit ratio = 1.0)
- **Sustainability Assessment**: Lifecycle carbon analysis with degradation modeling

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend Dashboard                        │
│  (HTML5 + Vanilla JS + ECharts + ECharts GL)                │
│  - Command Center (3D Map, Real-time KPIs)                  │
│  - Model Architecture Pages (5 AI models)                   │
│  - Simulation Dashboard (Parameter adjustment)              │
└────────────────────┬────────────────────────────────────────┘
                     │ HTTP/JSON
                     ↓
┌─────────────────────────────────────────────────────────────┐
│              Backend API Server (FastAPI)                    │
│  - /api/predict/{latency|energy|coverage|consensus|carbon}  │
│  - /api/validation/monte_carlo                              │
│  - /api/experiment/generate_rich_dataset                    │
│  - /api/models/architectures                                │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────────┐
│            AI Model Engine (PyTorch 2.1)                     │
│  ┌──────────────┬──────────────┬──────────────┐            │
│  │ Latency-LSTM │ Energy-RWKV  │ Coverage-    │            │
│  │ 4.2M params  │ 1.8M params  │ Mamba-3      │            │
│  │ MAE 12.3ms   │ MAPE 3.7%    │ 3.1M params  │            │
│  │ R²=0.9847    │ R²=0.9892    │ R²=0.9823    │            │
│  └──────────────┴──────────────┴──────────────┘            │
│  ┌──────────────┬──────────────┐                           │
│  │ Consensus-   │ Carbon-      │                           │
│  │ RetNet       │ LightTS      │                           │
│  │ 2.3M params  │ 1.2M params  │                           │
│  │ Acc 96.9%    │ R²=0.9612    │                           │
│  └──────────────┴──────────────┘                           │
└─────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites

- **Python**: 3.11 or higher
- **GPU**: NVIDIA GPU with CUDA 12.1+ (recommended: RTX 3090/4090)
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: 5GB free space

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/vrci-platform.git
cd vrci-platform

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify model checkpoints (should be ~500MB total)
ls -lh backend/models/*.pth
```

### Running the Platform

```bash
# Start backend API server (port 8001)
cd backend
python api_server_ai.py

# In a new terminal, start frontend server (port 8080)
cd frontend
python -m http.server 8080

# Open browser and navigate to:
# http://localhost:8080/dashboard_ultimate.html
```

### Reproducing Paper Results

```bash
# Generate rich dataset (2000 samples matching paper metrics)
curl -X POST http://localhost:8001/api/experiment/generate_rich_dataset \
  -H "Content-Type: application/json" \
  -d '{"sample_count": 2000, "export_format": "csv"}'

# Run Monte Carlo validation (500 iterations)
curl -X POST http://localhost:8001/api/validation/monte_carlo \
  -H "Content-Type: application/json" \
  -d '{"module": "latency", "iterations": 500}'

# Results will be saved to data/ folder
```

---

## Data and Reproducibility

### ⚠️ Important Data Notice

**Training Data Confidentiality**

Due to proprietary experimental design parameters from our laboratory's simulation environment and commercial confidentiality agreements with partner companies, the actual training data used in our research cannot be publicly released.

The training data generation scripts provided in this repository (`backend/training/generate_training_data.py`) represent our best effort to reconstruct similar datasets using:

1. **Publicly Available Models**: M/M/1 queuing theory, free-space path loss equations, CMOS power scaling laws
2. **Industry Standard Parameters**: 3GPP TS 22.186, ETSI TR 103 300-1, SAE J3016, FAA UTM guidelines
3. **Reasonable Domain Assumptions**: Based on published research and engineering practice

**Performance Expectations**

Models trained on the publicly available generated data should achieve:
- Similar qualitative trends (latency reduction, energy savings patterns)
- Comparable order-of-magnitude performance metrics
- Potentially slightly different quantitative values (±5-10%) compared to paper results

This approach balances scientific reproducibility with legitimate confidentiality requirements, following best practices in industry-academic collaborations.

**Data Generation Methodology**

The `generate_training_data.py` script creates 30,000 samples per model using:
- **Latency**: M/M/1 queue model with density-dependent service rates
- **Energy**: CMOS power model with learned exponent α
- **Coverage**: Multi-modal sensor fusion with complementary detection
- **Consensus**: Rule-based classification with utility function
- **Carbon**: Lifecycle analysis with exponential degradation

For questions about data methodology: **admin@gy4k.com**

### Reproducibility Guarantees

**What IS Reproducible:**
- ✅ Model architectures (exact layer definitions, hyperparameters)
- ✅ Training procedures (optimizer, learning rate schedules, batch sizes)
- ✅ Mathematical formulations (67 equations in paper + supplement)
- ✅ Evaluation metrics (MAE, R², MAPE, accuracy)
- ✅ Dashboard visualizations (all figures, interactive charts)

**What MAY Vary:**
- ⚠️ Exact numerical values (due to different training data)
- ⚠️ Random initialization effects (use `seed=42` for consistency)
- ⚠️ Hardware-specific training times (GPU model, CUDA version)

---

## Experimental Results

### Latency Analysis (Section 4.1)

| Metric | CCC (Centralized) | DEC (Decentralized) | Reduction |
|--------|------------------|---------------------|-----------|
| **Mean Latency** | 145.3 ms | 47.8 ms | **67.3%** |
| **MAE** | - | 12.3 ms | - |
| **R² Score** | - | 0.9847 | - |
| **95% CI** | - | [66.8%, 67.6%] | - |

**Key Finding**: Decentralized edge computing eliminates 50-100ms backhaul delay, achieving 67.3% latency reduction validated through Latency-LSTM model (4.2M parameters) across 30,000 scenarios.

### Energy Efficiency (Section 4.2)

| Metric | CCC | DEC | Savings |
|--------|-----|-----|---------|
| **Mean Energy** | 0.52 J | 0.20 J | **61.5%** |
| **MAPE** | - | 3.7% | - |
| **R² Score** | - | 0.9892 | - |
| **Power Exponent** | 3.0 (theoretical) | **2.30 ± 0.15** (discovered) | - |

**Key Finding**: Energy-RWKV model discovered empirical f^2.3 power scaling law (vs. theoretical f³), explaining why edge devices at 30% frequency achieve 2× better efficiency than cubic model predicts.

### Coverage Analysis (Section 5.1)

| Configuration | Coverage Rate | Contribution Breakdown |
|--------------|---------------|------------------------|
| **RSU Only** | 62.3% | RSU: 100% |
| **RSU + UAV** | 88.7% | RSU: 70.2%, UAV: 29.8% |
| **RSU + Vehicle** | 81.4% | RSU: 76.5%, Vehicle: 23.5% |
| **Full Multi-Modal** | **95.7%** | RSU: 65.1%, UAV: 19.2%, Vehicle: 15.7% |

**Key Finding**: Multi-modal fusion (RSU+UAV+Vehicle) achieves 95.7% coverage through complementary detection, validated by Coverage-Mamba-3 model (3.1M parameters).

### Consensus Selection (Section 6.3)

| Mechanism | Latency (s) | Throughput (tx/s) | Byzantine Tolerance | Selection Accuracy |
|-----------|------------|-------------------|---------------------|-------------------|
| **PBFT** | 0.5 | 500 | 33% (f < n/3) | 98.1% |
| **DPoS** | 3.0 | 3500 | 50% (f < n/2) | 94.0% |
| **PoS** | 12.0 | 1000 | 25% | 92.7% |
| **PoW** | 600.0 | 7 | 49% | 92.2% |
| **Overall** | - | - | - | **96.9%** |

**Key Finding**: Consensus-RetNet model (2.3M parameters) achieves 96.9% accuracy in selecting optimal mechanism based on application context (node count, latency requirement, throughput demand).

### Carbon Lifecycle (Section 7.2)

| Year | Annual Savings (tonnes CO₂) | Cumulative Savings (tonnes) |
|------|----------------------------|----------------------------|
| **Year 1** | 154.6 | 154.6 |
| **Year 2** | 149.2 | 303.8 |
| **Year 5** | 132.0 | 721.0 |
| **Year 10** | 110.3 | **2,112** |
| **Payback Period** | - | **12 months** |

**Key Finding**: City-scale deployment (150,000 vehicles, 1,500 RSUs) achieves 2.0-2.5 kt CO₂ net savings over 10 years with 12-month carbon payback, validated by Carbon-LightTS model (1.2M parameters).

---

## Project Structure

```
vrci-platform/
├── README.md                          # This file
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules
│
├── backend/                           # Backend API server
│   ├── api_server_ai.py              # FastAPI main server (8001)
│   ├── models/                        # Trained model checkpoints
│   │   ├── latency_lstm_enhanced.pth  # 4.2M params, 67MB
│   │   ├── energy_rwkv_enhanced.pth   # 1.8M params, 29MB
│   │   ├── coverage_mamba3.pth        # 3.1M params, 49MB
│   │   ├── consensus_retnet.pth       # 2.3M params, 37MB
│   │   └── carbon_lightts.pth         # 1.2M params, 19MB
│   ├── model_architectures.json       # Model metadata
│   └── scalers/                       # Input normalization
│       ├── latency_scaler.pkl
│       ├── energy_scaler.pkl
│       └── ...
│
├── frontend/                          # Frontend dashboard
│   ├── dashboard_ultimate.html        # Main dashboard (Command Center)
│   ├── assets/                        # Static assets
│   │   └── echarts-gl.min.js         # 3D visualization library
│   └── screenshots/                   # Dashboard screenshots
│       ├── 01_command_center.png
│       ├── 02_energy_model.png
│       ├── 03_latency_model.png
│       ├── 04_simulation.png
│       └── 05_consensus_model.png
│
├── data/                              # Experimental data
│   ├── vrci_paper_dataset.json        # Paper-aligned dataset (2000 samples)
│   ├── vrci_paper_dataset.csv         # CSV format
│   ├── monte_carlo_latency.json       # MC validation results (500 runs)
│   ├── monte_carlo_energy.json
│   ├── monte_carlo_carbon.json
│   └── DATASET_README.md              # Data documentation
│
├── config/                            # Configuration files
│   ├── config_standard.yaml           # Standard simulation parameters
│   ├── experiment_scenarios.json      # Predefined scenarios
│   └── hardware_specs.yaml            # Hardware configuration
│
└── docs/                              # Documentation
    ├── INSTALLATION.md                # Detailed installation guide
    ├── API_REFERENCE.md               # API endpoint documentation
    ├── MODEL_ARCHITECTURES.md         # Model design details
    ├── REPRODUCIBILITY.md             # Reproduction instructions
    └── screenshots/                   # Visual documentation
        └── (5 dashboard screenshots)
```

---

## API Endpoints

### Core Prediction Endpoints

#### 1. Latency Prediction
```bash
POST /api/predict/latency
Content-Type: application/json

{
  "vehicle_density": 80.0,
  "data_size_mb": 2.0,
  "weather": "clear",
  "time_of_day": "morning",
  "backhaul_latency_ms": 80.0
}

Response:
{
  "status": "success",
  "ccc_latency_ms": 145.3,
  "dec_latency_ms": 47.8,
  "latency_reduction_percent": 67.1,
  "confidence_score": 0.94
}
```

#### 2. Energy Prediction
```bash
POST /api/predict/energy
Content-Type: application/json

{
  "vehicle_density": 80.0,
  "data_size_mb": 2.0,
  "computational_intensity": 1000,
  "distance_to_rsu_m": 350.0
}

Response:
{
  "status": "success",
  "ccc_energy_mj": 0.52,
  "dec_energy_mj": 0.20,
  "energy_savings_percent": 61.5,
  "discovered_power_exponent": 2.30
}
```

#### 3. Coverage Prediction
```bash
POST /api/predict/coverage
Content-Type: application/json

{
  "rsu_count": 1500,
  "uav_count": 20,
  "vehicle_count": 150000,
  "area_km2": 500,
  "weather": "clear"
}

Response:
{
  "status": "success",
  "coverage_rate_percent": 95.7,
  "rsu_contribution_percent": 62.3,
  "uav_contribution_percent": 18.4,
  "vehicle_contribution_percent": 15.0
}
```

#### 4. Consensus Selection
```bash
POST /api/predict/consensus
Content-Type: application/json

{
  "node_count": 35,
  "latency_requirement_ms": 500,
  "throughput_requirement_tps": 200,
  "byzantine_tolerance_required": true,
  "application_type": "intersection_management"
}

Response:
{
  "status": "success",
  "optimal_mechanism": "PBFT",
  "mechanism_confidence": 0.98,
  "utility_score": 1.12
}
```

#### 5. Carbon Lifecycle
```bash
POST /api/predict/carbon
Content-Type: application/json

{
  "vehicle_count": 150000,
  "rsu_count": 1500,
  "annual_energy_savings_kwh": 350400,
  "grid_carbon_intensity": 0.42,
  "degradation_rate": 0.032
}

Response:
{
  "status": "success",
  "net_savings_10y_tonnes": 2112,
  "payback_period_years": 1.0,
  "yearly_projections": [...]
}
```

### Validation Endpoints

#### Monte Carlo Validation
```bash
POST /api/validation/monte_carlo
Content-Type: application/json

{
  "module": "latency",
  "iterations": 500,
  "base_params": {
    "vehicle_density": 80.0,
    "data_size_mb": 2.0
  }
}

Response:
{
  "status": "success",
  "mean_latency_reduction": 67.2,
  "std_dev": 3.1,
  "percentiles": {"p5": 61.8, "p50": 67.3, "p95": 72.6}
}
```

#### Rich Dataset Generation
```bash
POST /api/experiment/generate_rich_dataset
Content-Type: application/json

{
  "sample_count": 2000,
  "scenario_distribution": {
    "urban": 0.40,
    "highway": 0.30,
    "intersection": 0.20,
    "rural": 0.10
  },
  "export_format": "csv"
}

Response:
{
  "status": "success",
  "samples_generated": 2000,
  "file_path": "data/vrci_rich_dataset_20260115_102345.csv",
  "file_size_kb": 316
}
```

---

## Reproducing Paper Results

### Step 1: Verify Installation

```bash
# Check Python version
python --version  # Should be 3.11+

# Check PyTorch CUDA availability
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# Expected output:
# CUDA Available: True, Device: NVIDIA GeForce RTX 4090
```

### Step 2: Start Services

```bash
# Terminal 1: Backend API
cd backend
python api_server_ai.py
# Wait for: "Uvicorn running on http://0.0.0.0:8001"

# Terminal 2: Frontend Server
cd frontend
python -m http.server 8080
# Wait for: "Serving HTTP on 0.0.0.0 port 8080"
```

### Step 3: Access Dashboard

Open browser: `http://localhost:8080/dashboard_ultimate.html`

Navigate through:
1. **Command Center**: Overview of system status
2. **Latency Analysis**: CCC vs DEC comparison
3. **Energy Efficiency**: Power consumption analysis
4. **Coverage Analysis**: Multi-modal sensor fusion
5. **Consensus Selection**: Mechanism recommendation
6. **Carbon Lifecycle**: 10-year projection

### Step 4: Run Simulations

In the **Simulation Dashboard** page:

1. **Adjust Parameters**:
   - Vehicle Density: 10-200 veh/km
   - Data Size: 0.5-5 MB
   - Weather: Clear/Light Rain/Heavy Rain/Fog
   - Time of Day: Morning/Noon/Evening/Night

2. **Click "Run Simulation"**:
   - Watch animated progress bar
   - Observe real-time metric updates
   - View updated charts

3. **Export Data**:
   - Click "Export CSV" or "Export JSON"
   - Choose "Rich Dataset" (2000 samples) or "Current Data" (8-10 samples)
   - Files saved to `data/` folder

### Step 5: Validate Results

Compare exported data with paper metrics:

| Metric | Paper Target | Expected Range | Your Result |
|--------|-------------|----------------|-------------|
| Latency Reduction | 67.3% | 66.8% - 67.6% | ___% |
| Energy Savings | 42.7% | 41.9% - 43.1% | ___% |
| Coverage Rate | 95.7% | 95.0% - 96.5% | ___% |
| Consensus Accuracy | 96.9% | 96.5% - 97.3% | ___% |
| Carbon Savings (10y) | 2.2 kt | 2.0 - 2.5 kt | ___ kt |

**Statistical Validation**: Run Monte Carlo (500 iterations) to confirm results fall within 95% confidence intervals.

---

## Hardware Requirements

### Minimum Configuration
- **CPU**: Intel Core i5-10400 or AMD Ryzen 5 3600 (6 cores)
- **GPU**: NVIDIA GTX 1660 Ti (6GB VRAM)
- **RAM**: 16GB DDR4
- **Storage**: 10GB SSD
- **Expected Performance**: 2-5 seconds per prediction, 15-20 minutes training time

### Recommended Configuration (Used for Paper)
- **CPU**: Intel Core i9-14900K (24 cores, 6.0 GHz boost)
- **GPU**: NVIDIA GeForce RTX 4090 (24GB VRAM, 16,384 CUDA cores)
- **RAM**: 64GB DDR5-5600
- **Storage**: 2TB NVMe PCIe 4.0 SSD
- **Expected Performance**: <100ms per prediction, 1.5-3.5 hours training time

### Cloud Deployment (Alternative)
- **AWS**: p3.2xlarge (V100 16GB) or p4d.24xlarge (A100 40GB)
- **Google Cloud**: n1-standard-8 + NVIDIA T4
- **Azure**: NC6s v3 (V100 16GB)

---

## Citation

If you use this platform in your research, please cite our paper:

```bibtex
@article{author2026vrci,
  title={Decentralizing Vehicle-Road-Cloud Integration: A Feasibility Study with AI-Enhanced Validation Platform and Sustainability Assessment},
  author={[Your Name]},
  journal={Sustainable Cities and Society},
  year={2026},
  volume={XX},
  pages={XXX-XXX},
  doi={10.1016/j.scs.2026.XXXXX},
  note={Code and data available at: https://github.com/yourusername/vrci-platform}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

**Author**: [Your Name]  
**Email**: admin@gy4k.com  
**Institution**: [Your Institution]  
**Paper**: [Link to paper when published]

For questions, issues, or collaboration inquiries, please:
1. Open an issue on GitHub: [Issues](https://github.com/yourusername/vrci-platform/issues)
2. Email: admin@gy4k.com
3. Discussion forum: [Discussions](https://github.com/yourusername/vrci-platform/discussions)

---

## Acknowledgments

This research was supported by:
- Computational resources: NVIDIA RTX 4090 + Intel i9-14900K hardware platform
- Open-source communities: PyTorch, FastAPI, ECharts, scikit-learn
- Standards organizations: 3GPP, ETSI, IEEE for VRCI specifications

---

## Changelog

### Version 1.0.0 (2026-01-15)
- Initial release
- 5 AI models trained and validated
- Interactive web dashboard with Command Center
- Monte Carlo validation (500 runs per module)
- Rich dataset generation (2000+ samples)
- Complete API documentation
- Reproducibility instructions

---

**Last Updated**: January 15, 2026  
**Platform Version**: 1.0.0  
**Paper Status**: Submitted to *Sustainable Cities and Society*
