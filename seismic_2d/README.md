# 2D Seismic Processing and Integration Module

## Overview

This module processes and analyzes 2D seismic data for integration with the main 3D seismic workflow. It provides:

- **Line Inventory**: Automatic scanning and cataloging of all 2D SEGY files
- **Quality Assessment**: Data quality metrics for each line
- **Well-to-Seismic Ties**: Identification of 2D lines nearest to wells
- **Basemap Generation**: Visual display of line locations and well positions
- **2D/3D Integration**: Framework for combining 2D regional data with 3D detailed interpretation

## Usage

### Command Line

```bash
# Full processing run
python seismic_2d_automation.py "path/to/2d_folder" -o "outputs"

# Quick inventory scan (faster, less detailed)
python seismic_2d_automation.py "path/to/2d_folder" --inventory

# With well header file
python seismic_2d_automation.py "path/to/2d_folder" --well-header "wells.xlsx"

# Using config file
python seismic_2d_automation.py -c config_default.json

# Display specific line
python seismic_2d_automation.py "path/to/2d_folder" --display-line "CH-78-102"
```

### Python API

```python
from seismic_2d_automation import Seismic2DConfig, Seismic2DAutomation

config = Seismic2DConfig(
    input_directory="path/to/2d_lines",
    output_dir="outputs",
    well_header_file="wells.xlsx"
)

automation = Seismic2DAutomation(config)
results = automation.run()

# Access results
print(f"Lines processed: {results.lines_processed}")
print(f"Well ties found: {len(results.well_ties)}")
```

## Data Requirements

### 2D SEGY Files
- Standard SEGY format (.sgy, .segy, .SGY, .SEGY)
- Supported naming convention: `SURVEY-LINE_PROCESSING_BLOCK.SGY`
- Example: `CH-78-102_MIG_B1410.SGY`

### Well Header (Optional)
- Excel (.xlsx) or CSV format
- Required columns: Well Name, X/Easting, Y/Northing
- Coordinates should be in the same CRS as seismic headers

## Outputs

| Output | Description |
|--------|-------------|
| `line_inventory.csv` | Complete inventory of all 2D lines |
| `well_line_ties.csv` | Well-to-line proximity analysis |
| `basemap_2d_lines.png` | Map showing line locations and wells |
| `inventory_summary.png` | Quality and statistics plots |
| `2d_seismic_results.json` | Complete results in JSON format |

## Quality Tiers

Lines are classified into quality tiers based on:

| Tier | Criteria |
|------|----------|
| GOOD | < 5% dead traces, > min_traces |
| FAIR | 5-10% dead traces |
| POOR | > 10% dead traces |
| EXCLUDED | < min_traces (100 default) |

## Integration with 3D Workflow

The 2D module integrates with the main seismic processing pipeline:

1. **Regional Framework**: Use 2D lines for basin-wide structural interpretation
2. **Velocity Validation**: Cross-validate velocity models between 2D and 3D
3. **Well Ties**: Tie wells to both 2D and 3D where coverage overlaps
4. **Composite Interpretation**: Extend 3D horizon picks into 2D regional coverage

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `line_prefix_filter` | ["CH-78"] | Only process lines matching prefix |
| `include_migrated` | true | Include migrated (MIG) lines |
| `include_stacked` | true | Include stacked (STK) lines |
| `apply_agc` | true | Apply AGC for display |
| `agc_window_ms` | 500 | AGC window length |
| `min_traces` | 100 | Minimum traces for valid line |
| `max_dead_trace_pct` | 10.0 | Dead trace threshold for FAIR quality |

## Author

Moses Ekene Obasi
University of Calabar, Nigeria
PhD Research - Seismic Interpretation and Reservoir Characterization
