"""
================================================================================
DEEP LEARNING INTERPRETATION MODULE
PhD Research - Bornu Chad Basin
================================================================================

Integrates state-of-the-art deep learning models for seismic interpretation:
- FaultSeg3D: 3D CNN fault detection (Wu et al., 2019)
- DeepSeismic: Facies classification (Microsoft)
- CNN-for-ASI: Automated seismic interpretation (Waldeland et al., 2018)
- seismic_deep_learning: Educational DL workflows (Wrona et al., 2021)

Author: Moses Ekene Obasi
Institution: University of Calabar, Nigeria
================================================================================
"""

from .dl_fault_detection import FaultDetector
from .dl_facies_classification import FaciesClassifier
from .dl_interpretation import AutomatedInterpreter
from .dl_integration import DeepLearningIntegration

__all__ = [
    'FaultDetector',
    'FaciesClassifier',
    'AutomatedInterpreter',
    'DeepLearningIntegration'
]

__version__ = '1.0.0'
