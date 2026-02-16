# Horizon Interpretation Automation v5.0

Production-ready region-growing horizon tracking for 3D seismic volumes.

## Features

- Region-growing horizon tracking algorithm
- Multi-scale seeding (well + grid)
- Formation-specific parameters
- Structural closure identification
- Dip-constrained propagation
- Fault-guided tracking (optional)
- JSON structured output for automation
- CLI, API, and webhook support

## Tracking Algorithm

Uses optimized region-growing with:
- **8-connected** propagation (diagonal neighbors)
- **Correlation-based** similarity matching
- **Dip constraint** for geological validity
- **Coherence filtering** for fault zones

## Quick Start

```bash
# Install dependencies
pip install numpy scipy matplotlib segyio scikit-image tqdm

# Run with SEG-Y file
python horizon_interpretation_automation.py input.segy -o outputs

# Run with config file
python horizon_interpretation_automation.py -c config_default.json

# Generate default config
python horizon_interpretation_automation.py --create-config my_config.json
```

## Tracking Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `grid_density` | 50 | Seed point spacing in traces |
| `search_window` | 30 | Samples to search (±120ms at 4ms) |
| `similarity_threshold` | 0.4 | Minimum correlation |
| `max_dip` | 85° | Maximum apparent dip |
| `max_iterations` | 200000 | Tracking iteration limit |
| `correlation_window` | 9 | Correlation window (samples) |

## Chad Basin Formations

Pre-configured formations for Bornu Basin:

| Formation | Typical TWT | Description |
|-----------|-------------|-------------|
| Top_Chad | 500 ms | Pliocene-Recent continental |
| Top_Kerri_Kerri | 1200 ms | Paleocene-Eocene fluvial |
| Top_Fika_Shale | 2000 ms | Turonian-Santonian marine (SOURCE) |
| Top_Gongila | 2800 ms | Cenomanian-Turonian (RESERVOIR) |
| Top_Bima | 3500 ms | Albian-Cenomanian (PRIMARY RESERVOIR) |
| Base_Bima | 4200 ms | Basement unconformity |

## Output

- `horizon_results.json` - Structured results with all metrics
- `config_used.json` - Configuration for reproducibility
- `figures/{horizon}_interpretation.png` - Time structure + coverage
- `figures/{horizon}_structure_map.png` - Structure map with contours
- `data/{horizon}_surface.npy` - NumPy horizon array

## Closure Identification

Automatic structural closure detection:
- Watershed segmentation
- Minimum amplitude threshold (100 ms default)
- Area filtering
- Ranked by structural relief

## Quality Metrics

| Coverage | Quality |
|----------|---------|
| >60% | Excellent |
| 40-60% | Good |
| 20-40% | Fair |
| <20% | Poor |

## OpenClaw Integration

```python
{
    "name": "horizon_interpretation",
    "description": "Track seismic horizons automatically",
    "parameters": {
        "seismic_file": {"type": "string", "required": True},
        "output_dir": {"type": "string"}
    },
    "execute": "python horizon_interpretation_automation.py \"{seismic_file}\" -o \"{output_dir}\""
}
```

## Author

Moses Ekene Obasi
PhD Research - University of Calabar
