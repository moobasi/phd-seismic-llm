# Dead Trace Detection and Removal Automation v5.0

Production-ready dead trace detection and removal for 3D seismic data.

## Features

- Multi-criteria adaptive detection
- DBSCAN spatial clustering
- Interpolation method comparison
- Cleaned SEGY output generation
- Full automation support (CLI, API, webhooks)
- Structured JSON results

## Quick Start

```bash
# Install dependencies
pip install numpy scipy matplotlib seaborn pandas segyio tqdm scikit-learn

# Run with SEGY file
python dead_trace_automation.py "path/to/file.segy" -o "cleaned.segy"

# Run with config file
python dead_trace_automation.py -c config_default.json
```

## Detection Criteria

1. **All zeros**: Complete data loss (all samples = 0)
2. **High zero content**: >95% of samples are zero
3. **Low variance**: Variance below adaptive threshold (μ - 3σ)
4. **Low RMS**: RMS amplitude below adaptive threshold

## Output

- `dead_trace_results.json` - Structured results
- `config_used.json` - Configuration for reproducibility
- `figures/detection_overview.png` - Visualization
- `*_cleaned.segy` - Cleaned SEGY file (dead traces removed)

## Methodology

Based on:
- Yilmaz (2001): Seismic Data Analysis, SEG
- Ross (2018): 3D Seismic Data Quality Control
- Ester et al. (1996): DBSCAN clustering algorithm

## OpenClaw Integration

Register as tool:
```python
{
    "name": "dead_trace_removal",
    "description": "Detect and remove dead traces from seismic data",
    "parameters": {
        "segy_file": {"type": "string", "required": True},
        "output_file": {"type": "string"}
    },
    "execute": "python dead_trace_automation.py \"{segy_file}\" -o \"{output_file}\""
}
```

## n8n Integration

Use HTTP Request node to call the API wrapper (see EDA folder for API pattern).

## Author

Moses Ekene Obasi
PhD Research - University of Calabar
