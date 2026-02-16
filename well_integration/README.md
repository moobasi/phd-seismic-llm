# Well Data Integration Automation v5.0

Production-ready well log analysis and velocity modeling for seismic-well ties.

## Features

- Multi-well LAS file processing
- Automated quality assessment and ranking
- Petrophysical calculations (Vsh, porosity, Sw, permeability)
- Formation identification
- Regional velocity model (Gaussian Process)
- Time-depth conversion functions
- JSON structured output for automation
- CLI, API, and webhook support

## Petrophysical Methods

| Property | Method | Reference |
|----------|--------|-----------|
| **Vshale** | Larionov (1969) | For Cretaceous rocks |
| **Porosity** | Density-Sonic average | Wyllie + Raymer-Hunt |
| **Sw** | Archie (1942) | a=1, m=2, n=2 |
| **Permeability** | Timur (1968) | From porosity and Swirr |
| **Density** | Gardner (1974) | Velocity-density relation |

## Quick Start

```bash
# Install dependencies
pip install numpy scipy matplotlib seaborn pandas lasio scikit-learn tqdm

# Run with LAS directory
python well_integration_automation.py "path/to/las_folder" -o "outputs"

# Run with config file
python well_integration_automation.py -c config_default.json

# Generate default config
python well_integration_automation.py --create-config my_config.json
```

## Well Quality Tiers

| Tier | Score | Description |
|------|-------|-------------|
| **TIER 1** | 80-100 | Excellent - all logs with >70% coverage |
| **TIER 2** | 60-79 | Good - DT and GR with good coverage |
| **TIER 3** | 40-59 | Usable - minimum data for analysis |
| **EXCLUDED** | 0-39 | Poor - insufficient data |

## Chad Basin Formations

The framework includes predefined formations:
- Chad Formation (0-1000m)
- Fika Shale (1000-2500m)
- Gongila Formation (2500-3500m)
- Bima Formation (3500-5000m)

## Output

- `well_results.json` - Structured results with all metrics
- `config_used.json` - Configuration for reproducibility
- `figures/quality_ranking.png` - Well quality ranking
- `figures/velocity_depth.png` - Velocity-depth profiles
- `figures/log_coverage.png` - Log coverage heatmap
- `data/velocity_*.txt` - Time-depth conversion functions
- `data/petrophysics_*.csv` - Complete well data
- `data/reservoir_summary.csv` - Reservoir intervals
- `velocity_model.pkl` - Trained velocity model

## Velocity Model

Uses Gaussian Process regression for:
- Interpolating velocity between wells
- Estimating uncertainty
- Cross-validation metrics (R2, RMSE)

## OpenClaw Integration

Register as tool:
```python
{
    "name": "well_integration",
    "description": "Process well logs for seismic-well ties",
    "parameters": {
        "las_directory": {"type": "string", "required": True},
        "output_dir": {"type": "string"}
    },
    "execute": "python well_integration_automation.py \"{las_directory}\" -o \"{output_dir}\""
}
```

## n8n Integration

Use HTTP Request node to call the API wrapper or execute via command node.

## Author

Moses Ekene Obasi
PhD Research - University of Calabar
