"""
VRCI Models Package
All 5 AI models for VRCI feasibility validation

Authors: NOK KO, Ma Zhiqin, Wei Zixian, Yu Changyuan
Contact: admin@gy4k.com
"""

from .latency_lstm_model import LatencyLSTMEnhanced, create_latency_model
from .energy_rwkv_model import EnergyRWKVEnhanced, create_energy_model
from .coverage_mamba_model import CoverageMamba3, create_coverage_model
from .consensus_retnet_model import ConsensusRetNet, create_consensus_model
from .carbon_lightts_model import CarbonLightTS, create_carbon_model

__all__ = [
    'LatencyLSTMEnhanced',
    'EnergyRWKVEnhanced',
    'CoverageMamba3',
    'ConsensusRetNet',
    'CarbonLightTS',
    'create_latency_model',
    'create_energy_model',
    'create_coverage_model',
    'create_consensus_model',
    'create_carbon_model'
]

__version__ = '1.0.0'
__author__ = 'VRCI Research Team'
__contact__ = 'admin@gy4k.com'
