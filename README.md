# LLM-Assisted Seismic Interpretation Framework

## PhD Research Project

**Title:** Development of an LLM-Assisted Seismic Interpretation Framework: Validation Using Bornu Chad Basin Data, Nigeria

**Candidate:** Moses Ekene Obasi
**Supervisor:** Prof. Dominic Akam Obi
**Institution:** Department of Geology, University of Calabar, Nigeria

---

## Overview

This research develops and validates an LLM-assisted seismic interpretation framework that integrates Large Language Model capabilities directly into the interpretation workflow. The framework provides:

- Intelligent analysis of interpretation results in geological context
- Visual interpretation of seismic maps, cross-sections, and attribute displays
- Decision guidance based on current results and objectives
- Natural language interface for accessible interpretation
- Automated documentation of interpretation workflow

## Project Structure

```
PHD REVIEW/
├── PhD_Proposal/           # Research proposal documents
├── eda/                    # Exploratory Data Analysis module
├── dead_trace/             # Dead trace detection and correction
├── horizon_interpretation/ # Automated horizon tracking
├── horizon_attributes/     # Seismic attribute extraction
├── well_integration/       # Well log integration
├── seismic_2d/            # 2D seismic processing
├── integration_2d3d/      # 2D-3D integration
├── inversion/             # Seismic inversion module
├── outputs/               # Processing results and figures
├── seismic_processor.py   # Main processing pipeline
├── pipeline_config.json   # Pipeline configuration
└── requirements.txt       # Python dependencies
```

## Validation Data

- **3D Seismic:** Post-stack volume from Bornu Chad Basin (~1,955 km²)
- **2D Seismic:** 62 regional lines
- **Well Logs:** 6 exploration wells with gamma ray, sonic, density, resistivity

## Installation

```bash
pip install -r requirements.txt
```

## Key Technologies

- **Python** for framework implementation
- **NumPy, SciPy** for numerical processing
- **Matplotlib** for visualization
- **segyio** for SEG-Y data handling
- **lasio** for LAS well log files
- **Claude API** for LLM integration

## License

This project is part of PhD research at the University of Calabar, Nigeria.

## Contact

- **Email:** me.obasi@gmail.com
- **GitHub:** [@moobasi](https://github.com/moobasi)
