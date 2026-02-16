# Seismic EDA Automation Framework v5.0

Production-ready Exploratory Data Analysis for large 3D seismic datasets.

## Features

- **Configuration-driven**: JSON config files for reproducibility
- **Structured output**: Machine-readable JSON results
- **n8n integration**: REST API and webhook support
- **Memory-efficient**: Streaming algorithms for large datasets
- **Caching**: Avoid recomputation with intelligent caching
- **Modular**: Enable/disable specific analyses
- **Publication-ready**: Comprehensive statistics and figures

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run with SEGY file
python seismic_eda_automation.py "path/to/file.segy" -o results/

# Run with config file
python seismic_eda_automation.py -c config_default.json

# Generate default config
python seismic_eda_automation.py --create-config my_config.json
```

## Output

The script generates:

1. **eda_results.json** - Complete structured results
2. **config_used.json** - Configuration for reproducibility
3. **figures/** - Visualization PNGs

## n8n Integration

See `N8N_INTEGRATION_GUIDE.md` for detailed instructions.

Quick setup:
```bash
# Start API server
python n8n_api_wrapper.py

# API available at http://localhost:8000
# Swagger docs at http://localhost:8000/docs
```

## Analysis Modules

| Module | Description | Output |
|--------|-------------|--------|
| Statistics | Amplitude statistics with normality tests | Mean, std, skewness, kurtosis |
| Quality | Dead traces, clipping, SNR analysis | Quality metrics, spatial maps |
| Spectral | Frequency content and bandwidth | Peak freq, bandwidth, decay |
| Resolution | Vertical and lateral resolution | Resolution estimates |
| Attributes | RMS, instantaneous, frequency attributes | Attribute statistics |
| Spatial | Quadrant analysis with ANOVA | Spatial patterns |

## Key Improvements over v4.0

1. **Automation-ready**: CLI, API, webhook support
2. **Structured JSON output**: For downstream processing
3. **Configurable**: External config files
4. **Caching**: Faster reruns
5. **Progress callbacks**: Real-time updates to n8n
6. **Modular pipeline**: Enable/disable analyses

## Files

- `seismic_eda_automation.py` - Main EDA framework
- `n8n_api_wrapper.py` - REST API for n8n
- `config_default.json` - Sample configuration
- `requirements.txt` - Python dependencies
- `N8N_INTEGRATION_GUIDE.md` - Integration guide

## Author

Moses Ekene Obasi
PhD Research - University of Calabar
