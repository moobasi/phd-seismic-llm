"""
Seismic Interpretation Package
==============================
PhD Research - University of Calabar

Modules:
- synthetic_seismogram: Well-to-seismic tie
- horizon_mapping: Horizon picking and structure mapping
- seismic_attributes: Attribute analysis and DHI detection
- run_interpretation: Master workflow
"""

from .synthetic_seismogram import SyntheticSeismogram
from .horizon_mapping import HorizonPicker, StructureMapper, DepthConverter
from .seismic_attributes import SeismicAttributes, DHIDetector
