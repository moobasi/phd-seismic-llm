# Horizon Attribute Analysis Automation v5.0

Memory-efficient horizon-based attribute extraction with DHI and prospect identification.

## Features

- Memory-efficient extraction (8GB RAM compatible)
- Auto-detect interpreted horizons or use fixed time slices
- Multi-attribute analysis (8 attributes)
- DHI (Direct Hydrocarbon Indicator) identification
- Automatic prospect ranking
- Well correlation analysis
- JSON structured output for automation
- CLI, API, and webhook support

## Attributes Supported

| Attribute | Description | DHI Indicator |
|-----------|-------------|---------------|
| **Envelope** | Instantaneous amplitude | High amplitude |
| **Frequency** | Instantaneous frequency | Low frequency |
| **Phase** | Instantaneous phase | Phase anomaly |
| **Sweetness** | Amplitude/frequency ratio | High sweetness |
| **Semblance** | Coherence/similarity | Fault detection |
| **Dip** | Structural dip angle | Structural mapping |
| **Azimuth** | Dip direction | Fracture analysis |
| **Structure** | Structural attribute | Closure mapping |

## Quick Start

```bash
# Install dependencies
pip install numpy scipy matplotlib pandas scikit-image

# Run with base directory
python horizon_attributes_automation.py "C:/path/to/attributes" -o outputs

# Run with config file
python horizon_attributes_automation.py -c config_default.json

# Generate default config
python horizon_attributes_automation.py --create-config my_config.json
```

## Memory Efficiency

The framework loads only required time slices instead of full 3D volumes:

| Mode | Full Volume | Slices Only | Reduction |
|------|-------------|-------------|-----------|
| Per Attribute | ~17 GB | ~50 MB | ~340x |
| Total (8 attrs) | ~136 GB | ~400 MB | ~340x |

## DHI Identification

DHI criteria for hydrocarbon detection:
- **High Amplitude**: >1.2 std above mean
- **Low Frequency**: <25 Hz
- **High Sweetness**: >0.3
- **Phase Anomaly**: >20° from background

Prospects require 3+ matching criteria.

## Output

- `attribute_results.json` - Structured results
- `config_used.json` - Configuration for reproducibility
- `statistics.json` - Per-attribute statistics
- `correlations.json` - Well correlation results
- `prospects.csv` - Ranked prospect table
- `{horizon}_attributes.png` - Multi-attribute maps
- `{horizon}_prospects.png` - DHI and prospect map

## Well Control

Pre-configured wells for Bornu Basin:
- HEREWA-01 (Excellent)
- MASU-1 (Excellent)
- KASADE-01 (Good)
- NGAMMA-EAST-1 (Fair)
- BULTE-1 (Excellent)
- NGOR-NORTH-1 (Good)

## OpenClaw Integration

```python
{
    "name": "horizon_attributes",
    "description": "Extract seismic attributes at horizons",
    "parameters": {
        "base_dir": {"type": "string", "required": True},
        "output_dir": {"type": "string"}
    },
    "execute": "python horizon_attributes_automation.py \"{base_dir}\" -o \"{output_dir}\""
}
```

## Author

Moses Ekene Obasi
PhD Research - University of Calabar
