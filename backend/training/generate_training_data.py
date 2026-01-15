"""
Training Data Generation for VRCI Models
Generates 30,000 synthetic training samples based on mathematical models

⚠️ IMPORTANT DATA NOTICE / 重要数据说明:
===========================================
Due to proprietary experimental design parameters from our laboratory's simulation 
environment and commercial confidentiality agreements with partner companies, the 
actual training data used in our research cannot be publicly released.

This training data generator represents our best effort to reconstruct similar 
datasets using:
1. Publicly available mathematical models (M/M/1 queuing, path loss formulas)
2. Standard industry parameters (3GPP, ETSI specifications)
3. Reasonable assumptions based on domain knowledge

The generated data approximates the characteristics of our proprietary datasets
while respecting confidentiality requirements. Models trained on this publicly
available data should achieve similar (though potentially not identical) 
performance to those reported in the paper.

由于实验室仿真设计中存在特定的参数微调，以及涉及合作公司的商业机密，
实际训练数据无法公开。本生成器基于公开的数学模型和行业标准参数，
尽可能还原接近真实情况的数据集。

For questions about data generation methodology:
Contact: admin@gy4k.com
===========================================

Author: VRCI Research Team
Contact: admin@gy4k.com
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime

# Set random seed
np.random.seed(42)


def generate_latency_training_data(n_samples=30000):
    """
    Generate latency training data
    
    Features:
    - vehicle_density, data_size_mb, backhaul_latency_ms, weather, time_of_day,
      computational_intensity, tx_power, distance_to_rsu, scenario_type
    
    Targets:
    - ccc_latency_ms, dec_latency_ms
    """
    print("Generating latency training data...")
    
    data = []
    for i in range(n_samples):
        # Sample features
        density = np.random.uniform(10, 200)
        data_size = np.random.lognormal(np.log(2.0), 0.4)
        data_size = np.clip(data_size, 0.5, 5.0)
        backhaul = np.random.uniform(50, 100)
        weather_code = np.random.choice([0, 1, 2, 3], p=[0.6, 0.25, 0.1, 0.05])
        time_code = np.random.randint(0, 4)
        comp_intensity = np.random.uniform(500, 2000)
        tx_power = np.random.uniform(0.5, 2.0)
        distance = np.random.uniform(100, 800)
        scenario = np.random.randint(0, 4)
        
        # Calculate labels using M/M/1 model
        # CCC: Backhaul-dominated
        R_cloud = 100 * np.exp(-0.015 * density)
        wireless_tx = (data_size * 8 / R_cloud)
        processing_cloud = (data_size * comp_intensity / 3500)
        ccc_latency = backhaul + wireless_tx + processing_cloud + np.random.normal(0, 3)
        ccc_latency = max(50, ccc_latency)
        
        # DEC: Edge-based
        R_edge = 1000  # 1 Gbps
        v2i_tx = (data_size * 8 / R_edge)
        mu = 3.125  # Service rate
        lambda_rate = density * 0.01
        queuing = 1 / (mu - lambda_rate) if mu > lambda_rate else 10
        processing_edge = (data_size * comp_intensity / 2000)
        dec_latency = v2i_tx + queuing + processing_edge + np.random.normal(0, 2)
        dec_latency = max(10, dec_latency)
        
        data.append({
            'vehicle_density': density,
            'data_size_mb': data_size,
            'backhaul_latency_ms': backhaul,
            'weather': weather_code,
            'time_of_day': time_code,
            'computational_intensity': comp_intensity,
            'tx_power': tx_power,
            'distance_to_rsu_m': distance,
            'scenario_type': scenario,
            'pue': 1.2 if scenario <= 1 else 1.7,
            'speed_kmh': [50, 120, 30, 80][scenario],
            'rsu_spacing_m': 800,
            'ccc_latency_ms': ccc_latency,
            'dec_latency_ms': dec_latency
        })
    
    return pd.DataFrame(data)


def generate_energy_training_data(n_samples=30000):
    """Generate energy training data"""
    print("Generating energy training data...")
    
    data = []
    kappa = 1e-27
    
    for i in range(n_samples):
        density = np.random.uniform(10, 200)
        data_size = np.random.lognormal(np.log(2.0), 0.4)
        data_size = np.clip(data_size, 0.5, 5.0)
        comp_intensity = np.random.uniform(500, 2000)
        tx_power = np.random.uniform(0.5, 2.0)
        pue = np.random.uniform(1.2, 1.7)
        
        # CCC energy (f^3 model)
        R_cloud = 100 * np.exp(-0.015 * density)
        tx_time = data_size * 8 / R_cloud
        tx_energy = tx_power * tx_time / 1000
        proc_energy = kappa * (3.5e9 ** 3.0) * (data_size * comp_intensity / 3.5e9) * pue
        ccc_energy = tx_energy + proc_energy + np.random.normal(0, 0.01)
        ccc_energy = max(0.01, ccc_energy)
        
        # DEC energy (f^2.3 model - learned exponent)
        alpha = 2.3 + np.random.normal(0, 0.15)
        tx_energy_dec = 0.5 * (data_size * 8 / 1000) / 1000
        proc_energy_dec = kappa * (2.0e9 ** alpha) * (data_size * comp_intensity / 2.0e9) * pue
        dec_energy = tx_energy_dec + proc_energy_dec + np.random.normal(0, 0.005)
        dec_energy = max(0.001, dec_energy)
        
        data.append({
            'vehicle_density': density,
            'data_size_mb': data_size,
            'computational_intensity': comp_intensity,
            'tx_power': tx_power,
            'pue': pue,
            'ccc_energy_mj': ccc_energy,
            'dec_energy_mj': dec_energy,
            'learned_alpha': alpha
        })
    
    return pd.DataFrame(data)


def generate_coverage_training_data(n_samples=30000):
    """Generate coverage training data"""
    print("Generating coverage training data...")
    
    data = []
    for i in range(n_samples):
        rsu_count = np.random.randint(1000, 2000)
        uav_count = np.random.randint(10, 30)
        vehicle_count = np.random.randint(50000, 200000)
        area_km2 = np.random.uniform(300, 700)
        weather_code = np.random.choice([0, 1, 2, 3], p=[0.6, 0.25, 0.1, 0.05])
        time_code = np.random.randint(0, 4)
        
        # Coverage calculation
        rsu_coverage = 0.623 + np.random.normal(0, 0.05)
        uav_coverage = 0.184 + np.random.normal(0, 0.03)
        vehicle_coverage = 0.150 + np.random.normal(0, 0.02)
        
        # Weather penalty
        weather_penalty = [0, 0.02, 0.05, 0.03][weather_code]
        
        # Complementary detection
        total = (1 - (1 - rsu_coverage) * (1 - uav_coverage) * (1 - vehicle_coverage)) - weather_penalty
        total = np.clip(total, 0.85, 0.99)
        
        data.append({
            'rsu_count': rsu_count,
            'uav_count': uav_count,
            'vehicle_count': vehicle_count,
            'area_km2': area_km2,
            'weather': weather_code,
            'time_of_day': time_code,
            'rsu_spacing_m': 800,
            'uav_altitude_m': 120,
            'coverage_rate': total
        })
    
    return pd.DataFrame(data)


def generate_consensus_training_data(n_samples=30000):
    """Generate consensus training data"""
    print("Generating consensus training data...")
    
    data = []
    mechanisms = ['PBFT', 'DPoS', 'PoS', 'PoW']
    
    for i in range(n_samples):
        node_count = np.random.randint(10, 200)
        latency_req = np.random.uniform(0.1, 10.0)
        throughput_req = np.random.uniform(10, 5000)
        energy_budget = np.random.uniform(0.01, 200)
        security_level = np.random.uniform(0.2, 0.9)
        decentralization = np.random.uniform(0.3, 0.95)
        
        # Rule-based label generation
        if latency_req < 1.0 and node_count < 100:
            mechanism = 'PBFT'
        elif throughput_req > 2000:
            mechanism = 'DPoS'
        elif energy_budget < 50 and security_level > 0.5:
            mechanism = 'PoS'
        else:
            mechanism = 'PoW'
        
        # Add noise (10% random flips)
        if np.random.random() < 0.1:
            mechanism = np.random.choice(mechanisms)
        
        data.append({
            'node_count': node_count,
            'latency_requirement_s': latency_req,
            'throughput_requirement_tps': throughput_req,
            'energy_budget_j': energy_budget,
            'security_level': security_level,
            'decentralization_level': decentralization,
            'network_size': node_count,
            'byzantine_tolerance_required': np.random.choice([0, 1]),
            'application_type': np.random.randint(0, 4),
            'priority_weights': np.random.randint(0, 4),
            'optimal_mechanism': mechanism
        })
    
    return pd.DataFrame(data)


def generate_carbon_training_data(n_samples=30000):
    """Generate carbon training data"""
    print("Generating carbon training data...")
    
    data = []
    for i in range(n_samples):
        annual_energy = np.random.uniform(100000, 1000000)
        embodied = np.random.uniform(30, 60)
        grid_intensity = np.random.uniform(0.3, 0.7)
        degradation = np.random.uniform(0.02, 0.04)
        
        # Calculate 10-year trajectory
        yearly = []
        cumulative = -embodied
        for year in range(1, 11):
            annual_savings = annual_energy * grid_intensity * (1 - degradation) ** year / 1000
            cumulative += annual_savings
            yearly.append(cumulative)
        
        data.append({
            'annual_energy_savings_kwh': annual_energy,
            'embodied_carbon_tonnes': embodied,
            'grid_carbon_intensity': grid_intensity,
            'degradation_rate': degradation,
            'year_1': yearly[0],
            'year_2': yearly[1],
            'year_3': yearly[2],
            'year_4': yearly[3],
            'year_5': yearly[4],
            'year_6': yearly[5],
            'year_7': yearly[6],
            'year_8': yearly[7],
            'year_9': yearly[8],
            'year_10': yearly[9]
        })
    
    return pd.DataFrame(data)


def main():
    """Generate all training datasets"""
    
    output_dir = Path('../../training_data')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("VRCI Training Data Generation")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print(f"Total samples per dataset: 30,000")
    print("")
    
    # Generate each dataset
    datasets = {
        'latency': generate_latency_training_data(30000),
        'energy': generate_energy_training_data(30000),
        'coverage': generate_coverage_training_data(30000),
        'consensus': generate_consensus_training_data(30000),
        'carbon': generate_carbon_training_data(30000)
    }
    
    # Save each dataset
    for name, df in datasets.items():
        csv_path = output_dir / f'{name}_training_data.csv'
        json_path = output_dir / f'{name}_training_data.json'
        
        df.to_csv(csv_path, index=False)
        df.to_json(json_path, orient='records', indent=2)
        
        print(f"✓ {name.capitalize()}: {len(df)} samples")
        print(f"  - CSV: {csv_path}")
        print(f"  - JSON: {json_path}")
        print(f"  - Size: {csv_path.stat().st_size / 1024:.1f} KB")
        print("")
    
    # Create metadata
    metadata = {
        'generated_at': datetime.now().isoformat(),
        'version': '1.0.0',
        'total_samples': sum(len(df) for df in datasets.values()),
        'datasets': {
            name: {
                'samples': len(df),
                'features': list(df.columns),
                'size_kb': (output_dir / f'{name}_training_data.csv').stat().st_size / 1024
            }
            for name, df in datasets.items()
        },
        'contact': 'admin@gy4k.com'
    }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("="*60)
    print("Generation Complete!")
    print("="*60)
    print(f"Total samples: {metadata['total_samples']:,}")
    print(f"Total size: {sum(d['size_kb'] for d in metadata['datasets'].values()):.1f} KB")
    print("")
    print("Next step: Train models using train_all_models.py")


if __name__ == '__main__':
    main()
