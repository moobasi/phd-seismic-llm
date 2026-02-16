# n8n Integration Guide for Seismic EDA Automation

## Overview

This guide explains how to integrate the Seismic EDA Automation framework with n8n for automated workflow processing.

## Integration Options

### Option 1: HTTP Request (Recommended for Production)

Run the API server and call it from n8n:

```bash
# Start the API server
python n8n_api_wrapper.py
```

In n8n, use an **HTTP Request** node:

```json
{
  "method": "POST",
  "url": "http://localhost:8000/eda/run-sync",
  "headers": {
    "Content-Type": "application/json"
  },
  "body": {
    "segy_file": "C:\\path\\to\\seismic.segy",
    "output_dir": "C:\\output\\path",
    "avg_velocity": 2500,
    "save_figures": true
  }
}
```

### Option 2: Execute Command (Simpler Setup)

Use n8n's **Execute Command** node:

```bash
python "C:\Users\moses\OneDrive\Documents\PHD REVIEW\eda\seismic_eda_automation.py" "{{$json.segy_file}}" -o "{{$json.output_dir}}" --webhook "{{$node.Webhook.json.headers.host}}/webhook/eda-progress"
```

### Option 3: Python Function (Advanced)

Use n8n's **Code** node with Python:

```python
import subprocess
import json

result = subprocess.run([
    'python',
    'C:\\path\\to\\seismic_eda_automation.py',
    items[0]['json']['segy_file'],
    '-o', items[0]['json']['output_dir']
], capture_output=True, text=True)

# Parse the JSON output
with open(items[0]['json']['output_dir'] + '/eda_results.json') as f:
    results = json.load(f)

return [{'json': results}]
```

## Sample n8n Workflows

### Basic EDA Workflow

```
[Webhook Trigger] → [Set SEGY Path] → [HTTP Request to EDA API] → [IF Quality Issues] → [Send Alert Email]
                                                                 → [Save Results to Database]
```

### Batch Processing Workflow

```
[Schedule Trigger] → [List SEGY Files] → [Loop Over Files] → [Run EDA] → [Aggregate Results] → [Generate Report]
```

### Quality Monitoring Workflow

```
[Watch Folder] → [New File Trigger] → [Run EDA] → [Check Dead Traces %] → [IF > 10%] → [Alert Slack]
                                                                         → [Log to Sheets]
```

## Webhook Progress Updates

The EDA script can send progress updates to n8n:

1. Create a **Webhook** node in n8n (e.g., `/webhook/eda-progress`)
2. Pass the webhook URL to the EDA script:

```bash
python seismic_eda_automation.py input.segy --webhook http://n8n:5678/webhook/eda-progress
```

Progress payload format:

```json
{
  "progress": 45.5,
  "phase": "Frequency Analysis",
  "message": "Processing: Frequency Analysis",
  "elapsed_seconds": 120.5,
  "timestamp": "2025-10-08T15:30:00"
}
```

## API Endpoints Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info and available endpoints |
| `/health` | GET | Health check |
| `/eda/run` | POST | Start async EDA job |
| `/eda/run-sync` | POST | Run EDA synchronously (wait for result) |
| `/eda/status/{job_id}` | GET | Get job status |
| `/eda/result/{job_id}` | GET | Get job results |
| `/eda/jobs` | GET | List all jobs |
| `/eda/download/{job_id}` | GET | Download results file |

## Request Body Schema

```json
{
  "segy_file": "string (required) - Path to SEGY file",
  "output_dir": "string - Output directory (default: eda_outputs)",
  "webhook_url": "string - Webhook for progress callbacks",
  "avg_velocity": "number - Average velocity in m/s (default: 2500)",
  "sample_traces": "number - Traces to sample (default: 30000)",
  "run_statistics": "boolean - Run statistics analysis (default: true)",
  "run_quality_checks": "boolean - Run quality checks (default: true)",
  "run_frequency_analysis": "boolean - Run frequency analysis (default: true)",
  "run_resolution_analysis": "boolean - Run resolution analysis (default: true)",
  "run_attribute_analysis": "boolean - Run attribute analysis (default: true)",
  "run_spatial_analysis": "boolean - Run spatial analysis (default: true)",
  "save_figures": "boolean - Generate figures (default: true)",
  "enable_cache": "boolean - Enable caching (default: true)"
}
```

## Response Schema

```json
{
  "status": "completed",
  "processing_time_seconds": 245.5,
  "results": {
    "timestamp": "2025-10-08T15:30:00",
    "survey": {...},
    "statistics": {...},
    "quality": {...},
    "spectral": {...},
    "resolution": {...},
    "attributes": {...},
    "spatial": {...},
    "recommendations": [...],
    "warnings": [...],
    "output_files": {...}
  }
}
```

## ClawBot / OpenClaw Integration

If using ClawBot (now OpenClaw), you can register this as a tool:

```python
# In your ClawBot/OpenClaw tool definition
{
    "name": "seismic_eda",
    "description": "Run comprehensive seismic EDA on a SEGY file",
    "parameters": {
        "segy_file": {"type": "string", "required": True},
        "output_dir": {"type": "string", "default": "eda_outputs"},
        "velocity": {"type": "number", "default": 2500}
    },
    "execute": "python seismic_eda_automation.py {segy_file} -o {output_dir}"
}
```

## Troubleshooting

### Common Issues

1. **SEGY file not found**: Ensure the path is correct and accessible
2. **Memory errors**: Reduce `sample_traces` in config
3. **Slow processing**: Enable caching with `enable_cache: true`
4. **Webhook not receiving**: Check n8n workflow is active and URL is correct

### Logs

Check logs in the output directory:
- `eda_results.json` - Full results
- `config_used.json` - Configuration used
- `figures/` - Generated visualizations

## Requirements

```bash
pip install numpy scipy matplotlib seaborn pandas segyio tqdm fastapi uvicorn requests
```
