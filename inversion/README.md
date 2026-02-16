# Seismic Inversion Automation v5.0

Model-based post-stack acoustic impedance inversion for reservoir characterization.

## Features

- Well log conditioning and QC
- Statistical wavelet extraction (3 methods)
- Low-frequency model construction
- Model-based acoustic impedance inversion
- Property prediction (porosity, lithology)
- Volumetric calculations (STOIIP)
- JSON structured output for automation
- CLI, API, and webhook support

## Methodology

Based on established inversion theory:
- **Russell & Hampson (1991)**: Post-stack inversion methods
- **Lancaster & Whitcombe (2000)**: Coloured inversion
- **Cooke & Cant (2010)**: Model-based inversion
- **Gardner et al. (1974)**: Velocity-density relationship

## Quick Start

```bash
# Install dependencies
pip install numpy scipy matplotlib segyio scikit-learn tqdm

# Run with SEG-Y file
python inversion_automation.py input.segy -o outputs

# Run with config file
python inversion_automation.py -c config_default.json

# With well data
python inversion_automation.py input.segy --wells well_data.json

# Generate default config
python inversion_automation.py --create-config my_config.json
```

## Wavelet Extraction Methods

| Method | Description | Best For |
|--------|-------------|----------|
| **least_squares** | Optimal L2 solution | General use (recommended) |
| **wiener** | Wiener deconvolution | Noisy data |
| **cross_correlation** | Simple correlation | Quick analysis |

## Inversion Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lambda_reg` | 0.1 | Regularization strength |
| `max_iterations` | 30 | Iterations per trace |
| `ai_min` | 1000 | Minimum AI bound (kg/m2/s) |
| `ai_max` | 20000 | Maximum AI bound (kg/m2/s) |

## Lithology Classification

Based on Bornu Basin calibration:
- **Sandstone** (Bima): AI < 8000 kg/m2/s
- **Shale** (Fika): 8000 < AI < 10000 kg/m2/s
- **Carbonate** (Gongila): AI > 10000 kg/m2/s

## Property Prediction

Porosity from acoustic impedance:
- Calibrated from well data if available
- Default: phi = -0.00004 * AI + 0.45

## Volumetrics (STOIIP)

Calculates Stock Tank Oil Initially In Place:
- Gross Rock Volume (GRV)
- Net Rock Volume (NRV = GRV * N/G)
- Hydrocarbon Pore Volume (HCPV)
- STOIIP = HCPV * (1-Sw) / Bo
- EUR = STOIIP * Recovery Factor

## Output

- `inversion_results.json` - Structured results
- `config_used.json` - Configuration for reproducibility
- `volumes/acoustic_impedance.npy` - AI volume
- `volumes/porosity.npy` - Porosity volume
- `volumes/lithology.npy` - Lithology volume
- `volumes/inversion_quality.npy` - Quality map
- `figures/inversion_qc_summary.png` - QC visualization

## Quality Metrics

| Metric | Description |
|--------|-------------|
| Mean Correlation | Average trace correlation |
| Quality % | Traces with r > 0.7 |
| AI Range | Min-max impedance |

## OpenClaw Integration

```python
{
    "name": "seismic_inversion",
    "description": "Run model-based acoustic impedance inversion",
    "parameters": {
        "seismic_file": {"type": "string", "required": True},
        "output_dir": {"type": "string"},
        "lambda_reg": {"type": "number", "default": 0.1}
    },
    "execute": "python inversion_automation.py \"{seismic_file}\" -o \"{output_dir}\" --lambda {lambda_reg}"
}
```

## Author

Moses Ekene Obasi
PhD Research - University of Calabar
