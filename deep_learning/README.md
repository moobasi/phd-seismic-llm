# Deep Learning Interpretation Module

## PhD Research - LLM-Assisted Seismic Interpretation Framework

This module integrates state-of-the-art deep learning models for automated seismic interpretation, feeding results to the LLM for intelligent geological interpretation.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DEEP LEARNING MODULE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ FaultSeg3D   │  │ DeepSeismic  │  │ CNN-for-ASI  │         │
│  │              │  │              │  │              │         │
│  │ 3D CNN Fault │  │ Facies       │  │ Texture      │         │
│  │ Detection    │  │ Classification│  │ Classification│         │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘         │
│         │                 │                 │                  │
│         └────────────────┼─────────────────┘                  │
│                          │                                     │
│                          ▼                                     │
│              ┌───────────────────────┐                        │
│              │   DL Integration      │                        │
│              │   - Result aggregation│                        │
│              │   - Risk assessment   │                        │
│              │   - Prospect ID       │                        │
│              └───────────┬───────────┘                        │
│                          │                                     │
└──────────────────────────┼─────────────────────────────────────┘
                           │
                           ▼
              ┌───────────────────────┐
              │   LLM ASSISTANT       │
              │   - Geological interp │
              │   - Recommendations   │
              │   - Report generation │
              └───────────────────────┘
```

## Components

### 1. Fault Detection (`dl_fault_detection.py`)
Based on **FaultSeg3D** (Wu et al., 2019, GEOPHYSICS)

- 3D U-Net architecture for fault segmentation
- Outputs fault probability volumes
- Identifies fault orientations and major fault systems

```python
from deep_learning import FaultDetector, FaultDetectionConfig

config = FaultDetectionConfig(seismic_file="data.segy")
detector = FaultDetector(config)
results = detector.detect_faults()
```

### 2. Facies Classification (`dl_facies_classification.py`)
Based on **Microsoft DeepSeismic** and **seismic_deep_learning** (Wrona et al., 2021)

- U-Net segmentation for seismic facies
- Multi-class classification (Channel Sand, Marine Shale, etc.)
- Basin-specific class definitions for Bornu Chad Basin

```python
from deep_learning import FaciesClassifier, FaciesClassificationConfig

config = FaciesClassificationConfig(seismic_file="data.segy")
classifier = FaciesClassifier(config)
results = classifier.classify_facies()
```

### 3. Automated Interpretation (`dl_interpretation.py`)
Based on **CNN-for-ASI** (Waldeland et al., 2018, The Leading Edge)

- Texture-based seismic classification
- Identifies continuous reflectors, chaotic zones, fault zones

```python
from deep_learning import AutomatedInterpreter, AutoInterpretationConfig

config = AutoInterpretationConfig(seismic_file="data.segy")
interpreter = AutomatedInterpreter(config)
results = interpreter.interpret()
```

### 4. Integration Module (`dl_integration.py`)
Master orchestrator that:

- Runs all DL models
- Aggregates results
- Performs fault-facies correlation
- Assesses reservoir risk
- Generates LLM interpretations
- Creates comprehensive reports

```python
from deep_learning import DeepLearningIntegration, DeepLearningConfig

config = DeepLearningConfig(seismic_file="data.segy")
integration = DeepLearningIntegration(config)
results = integration.run_full_interpretation()

# Get LLM-powered geological interpretation
print(results.llm_geological_interpretation)
```

## Installation

### Requirements
```bash
pip install torch torchvision  # PyTorch
pip install numpy scipy matplotlib
pip install segyio  # Seismic data I/O
pip install tqdm  # Progress bars
```

### GPU Support (Recommended)
```bash
# For CUDA 12.x
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.x
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Usage

### Command Line
```bash
# Full interpretation
python -m deep_learning.dl_integration data.segy --output-dir results

# Fault detection only
python -m deep_learning.dl_fault_detection data.segy --output-dir faults

# Facies classification only
python -m deep_learning.dl_facies_classification data.segy --output-dir facies
```

### Integration with Main Pipeline
The deep learning module is integrated as **Step 9** in the main seismic processing pipeline:

```bash
python seismic_processor.py --step 9
```

## Output Files

```
dl_interpretation/
├── faults/
│   ├── fault_probability.npy      # Fault probability volume
│   ├── fault_binary.npy           # Binary fault volume
│   ├── fault_detection_results.json
│   └── fault_detection_*.png      # Visualizations
├── facies/
│   ├── facies_classification.npy  # Classification volume
│   ├── facies_probabilities.npy   # Class probabilities
│   ├── facies_classification_results.json
│   └── facies_*.png               # Visualizations
├── combined_interpretation.png    # Summary figure
├── interpretation_report.md       # Comprehensive report
└── integrated_results.json        # All results in JSON
```

## Pre-trained Models

For production use, download pre-trained models:

1. **FaultSeg3D weights**: [GitHub - xinwucwp/faultSeg](https://github.com/xinwucwp/faultSeg)
2. **DeepSeismic weights**: [GitHub - microsoft/seismic-deeplearning](https://github.com/microsoft/seismic-deeplearning)
3. **CNN-for-ASI weights**: Contact authors (Waldeland et al.)

Place model files in `models/` directory and update config:
```python
config.model_path = "models/faultseg3d_weights.pth"
```

## References

1. Wu, X., Liang, L., Shi, Y., & Fomel, S. (2019). FaultSeg3D: Using synthetic data sets to train an end-to-end convolutional neural network for 3D seismic fault segmentation. *Geophysics*, 84(3), IM35-IM45.

2. Microsoft DeepSeismic Team. (2020). Deep learning for seismic interpretation on Azure. *GitHub Repository*.

3. Waldeland, A. U., Jensen, A. C., Gelius, L., & Solberg, A. H. S. (2018). Convolutional Neural Networks for Automated Seismic Interpretation. *The Leading Edge*, 37(7), 529-537.

4. Wrona, T., Pan, I., Bell, R., Gawthorpe, R. L., Fossen, H., & Brune, S. (2021). Deep learning tutorials for seismic interpretation. *DOI: 10.5880/GFZ.2.5.2021.001*

## Author

Moses Ekene Obasi
PhD Research - University of Calabar, Nigeria
Supervisor: Prof. Dominic Akam Obi
