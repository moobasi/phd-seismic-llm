"""
================================================================================
SEISMIC AI ASSISTANT v2.0 - Ollama-Powered Interactive GUI
================================================================================

PhD Research - Bornu Chad Basin
Author: Moses Ekene Obasi
Institution: University of Calabar, Nigeria

Enhanced Features:
- Live processing output with progress tracking
- Data input configuration (3D/2D seismic, wells, logs)
- Report generation (text, markdown)
- Image interpretation using Llava vision model
- Interactive chat for seismic questions

Usage:
    python seismic_ai_assistant_v2.py
================================================================================
"""

import os
import sys
import json
import threading
import subprocess
import queue
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any
import base64
import re

# Base directory
BASE_DIR = Path(__file__).parent.resolve()

# Import centralized configuration
try:
    from project_config import get_config, ProjectConfig
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

# Import fluid analysis module
try:
    from fluid_analysis import FluidAnalyzer
    FLUID_ANALYSIS_AVAILABLE = True
except ImportError:
    FLUID_ANALYSIS_AVAILABLE = False
    FluidAnalyzer = None

# Import seismic viewer module
try:
    from seismic_viewer import SeismicViewerGUI, WellTiePanel, SeismicDataLoader, WellTieValidator
    SEISMIC_VIEWER_AVAILABLE = True
except ImportError:
    SEISMIC_VIEWER_AVAILABLE = False
    SeismicViewerGUI = None
    WellTiePanel = None

# =============================================================================
# OLLAMA CLIENT WITH VISION SUPPORT
# =============================================================================

class OllamaClient:
    """Ollama API client with vision model support"""

    def __init__(self, model: str = "qwen3:32b"):
        self.model = model
        self.vision_model = "llava:13b"  # For image interpretation
        self.available_models = []
        self.is_connected = False
        self.preferred_models = ["qwen3:32b", "dolphin-mixtral:8x7b", "llava:13b"]

    def check_connection(self) -> bool:
        """Check if Ollama is running"""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                self.is_connected = True
                lines = result.stdout.strip().split('\n')
                self.available_models = []
                for line in lines[1:]:
                    if line.strip():
                        model_name = line.split()[0]
                        self.available_models.append(model_name)

                # Set best available model
                model_set = False
                for preferred in self.preferred_models:
                    if preferred in self.available_models:
                        self.model = preferred
                        model_set = True
                        break
                if not model_set and self.available_models:
                    # Skip embedding and vision models for chat
                    for m in self.available_models:
                        if "embed" not in m.lower() and "llava" not in m.lower():
                            self.model = m
                            break

                # Check for vision model
                self.has_vision = any("llava" in m.lower() for m in self.available_models)
                return True
        except Exception as e:
            print(f"Ollama connection error: {e}")
        self.is_connected = False
        return False

    def chat(self, prompt: str, system_prompt: str = "", context: str = "") -> str:
        """Send a chat request to Ollama"""
        if not self.is_connected:
            return "Error: Ollama is not connected. Please start Ollama first."

        full_prompt = ""
        if system_prompt:
            full_prompt += f"System: {system_prompt}\n\n"
        if context:
            full_prompt += f"Context:\n{context}\n\n"
        full_prompt += f"User: {prompt}"

        try:
            result = subprocess.run(
                ["ollama", "run", self.model, full_prompt],
                capture_output=True,
                text=True,
                timeout=180
            )
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                return f"Error: {result.stderr}"
        except subprocess.TimeoutExpired:
            return "Error: Request timed out. The model may be processing a complex query."
        except Exception as e:
            return f"Error: {str(e)}"

    def interpret_image(self, image_path: str, prompt: str = "Describe this seismic image in detail") -> str:
        """Interpret an image using the vision model (llava)"""
        if not self.is_connected:
            return "Error: Ollama is not connected."

        if not any("llava" in m.lower() for m in self.available_models):
            return "Error: Llava vision model not installed. Run: ollama pull llava:13b"

        # For llava, we need to use the API or a different approach
        # Using subprocess with image path
        try:
            # Llava can take image path directly in newer versions
            full_prompt = f"You are a seismic interpretation expert. {prompt}"

            result = subprocess.run(
                ["ollama", "run", self.vision_model, full_prompt, image_path],
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode == 0:
                return result.stdout.strip()
            else:
                # Try alternative method using base64
                return self._interpret_image_base64(image_path, prompt)
        except Exception as e:
            return f"Error interpreting image: {str(e)}"

    def _interpret_image_base64(self, image_path: str, prompt: str) -> str:
        """Alternative image interpretation using base64 encoding"""
        try:
            import requests

            # Read and encode image
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode()

            # Determine image type
            ext = Path(image_path).suffix.lower()
            mime_type = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".gif": "image/gif"
            }.get(ext, "image/png")

            # Send to Ollama API
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.vision_model,
                    "prompt": f"You are a seismic interpretation expert analyzing geophysical data. {prompt}",
                    "images": [image_data],
                    "stream": False
                },
                timeout=300
            )

            if response.status_code == 200:
                return response.json().get("response", "No response received")
            else:
                return f"API Error: {response.status_code}"

        except ImportError:
            return "Error: 'requests' library needed for image interpretation. Install with: pip install requests"
        except Exception as e:
            return f"Error: {str(e)}"


# =============================================================================
# STATE MANAGER
# =============================================================================

class StateManager:
    """Persistent state management"""

    def __init__(self):
        self.state_file = BASE_DIR / "processing_state.json"
        self.state = self._load_state()

    def _load_state(self) -> Dict:
        default_state = {
            "completed_steps": [],
            "step_timestamps": {},
            "step_results": {},
            "current_project": "",
            "last_updated": "",
            "notes": [],
            "data_paths": {
                "segy_3d": "",
                "segy_2d_folder": "",
                "well_folder": "",
                "well_header": "",
                "output_folder": str(BASE_DIR / "outputs")
            }
        }

        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    loaded = json.load(f)
                    default_state.update(loaded)
            except:
                pass

        return default_state

    def save(self):
        self.state["last_updated"] = datetime.now().isoformat()
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)

    def mark_completed(self, step_num: int, result_summary: str = ""):
        if step_num not in self.state["completed_steps"]:
            self.state["completed_steps"].append(step_num)
        self.state["step_timestamps"][str(step_num)] = datetime.now().isoformat()
        if result_summary:
            self.state["step_results"][str(step_num)] = result_summary
        self.save()

    def is_completed(self, step_num: int) -> bool:
        return step_num in self.state["completed_steps"]

    def get_next_step(self) -> int:
        for i in range(1, 9):
            if i not in self.state["completed_steps"]:
                return i
        return 0

    def reset_step(self, step_num: int):
        if step_num in self.state["completed_steps"]:
            self.state["completed_steps"].remove(step_num)
        self.save()

    def reset_all(self):
        self.state["completed_steps"] = []
        self.state["step_timestamps"] = {}
        self.state["step_results"] = {}
        self.save()

    def set_data_path(self, key: str, path: str):
        self.state["data_paths"][key] = path
        self.save()

    def get_data_path(self, key: str) -> str:
        return self.state.get("data_paths", {}).get(key, "")


# =============================================================================
# KNOWLEDGE BASE
# =============================================================================

class SeismicKnowledgeBase:
    """Knowledge base for seismic interpretation"""

    def __init__(self):
        self.outputs_dir = BASE_DIR / "outputs"
        self.well_outputs_dir = BASE_DIR / "well_outputs"
        self.knowledge_file = BASE_DIR / "seismic_knowledge_base.json"
        self.geological_knowledge = self._load_geological_knowledge()

        # Initialize fluid analyzer
        self.fluid_analyzer = None
        if FLUID_ANALYSIS_AVAILABLE:
            well_data_dir = self.well_outputs_dir / "data"
            if well_data_dir.exists():
                self.fluid_analyzer = FluidAnalyzer(str(well_data_dir))
                self.fluid_analyzer.load_all_wells()

        # Coordinate transformation
        self.coord_params = {
            "inline_range": [5047, 6047],
            "xline_range": [4885, 7020],
            "cdp_x_range": [325000.0, 377000.0],
            "cdp_y_range": [1420000.0, 1460000.0],
            "utm_zone": "33N"
        }

    def _load_geological_knowledge(self) -> Dict:
        if self.knowledge_file.exists():
            try:
                with open(self.knowledge_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {}

    def inline_xline_to_utm(self, inline: int, xline: int) -> tuple:
        il_range = self.coord_params["inline_range"]
        xl_range = self.coord_params["xline_range"]
        x_range = self.coord_params["cdp_x_range"]
        y_range = self.coord_params["cdp_y_range"]

        il_norm = (inline - il_range[0]) / (il_range[1] - il_range[0])
        xl_norm = (xline - xl_range[0]) / (xl_range[1] - xl_range[0])

        utm_x = x_range[0] + il_norm * (x_range[1] - x_range[0])
        utm_y = y_range[0] + xl_norm * (y_range[1] - y_range[0])

        return utm_x, utm_y

    def load_all_results(self) -> str:
        context_parts = []

        # Try to load various result files
        result_files = [
            ("eda_results.json", "3D SEISMIC EDA"),
            ("dead_trace_results.json", "DEAD TRACE ANALYSIS"),
            ("well_results.json", "WELL INTEGRATION"),
            ("horizon_results.json", "HORIZON INTERPRETATION"),
            ("attribute_results.json", "SEISMIC ATTRIBUTES"),
        ]

        for filename, title in result_files:
            filepath = self.outputs_dir / filename
            if not filepath.exists():
                filepath = self.well_outputs_dir / filename

            if filepath.exists():
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    context_parts.append(f"\n=== {title} ===")
                    # Summarize key findings
                    context_parts.append(json.dumps(data, indent=2)[:2000])
                except:
                    pass

        return "\n".join(context_parts) if context_parts else "No processing results available yet."

    def get_drilling_context(self) -> str:
        context = self.load_all_results()

        # Add geological knowledge
        if self.geological_knowledge:
            context += "\n\n=== GEOLOGICAL KNOWLEDGE BASE ===\n"
            context += json.dumps(self.geological_knowledge, indent=2)[:3000]

        return context

    def calculate_stoiip(self, area_km2: float, thickness_m: float, porosity: float = 0.20,
                         sw: float = 0.25, bo: float = 1.2, ntg: float = 0.6) -> Dict:
        area_m2 = area_km2 * 1e6
        grv_m3 = area_m2 * thickness_m
        nrv_m3 = grv_m3 * ntg
        hcpv_m3 = nrv_m3 * porosity * (1 - sw)
        stoiip_bbl = (hcpv_m3 * 6.2898) / bo
        eur_bbl = stoiip_bbl * 0.35

        return {
            "grv_m3": grv_m3,
            "nrv_m3": nrv_m3,
            "hcpv_m3": hcpv_m3,
            "stoiip_bbl": stoiip_bbl,
            "stoiip_mmstb": stoiip_bbl / 1e6,
            "eur_bbl": eur_bbl,
            "eur_mmstb": eur_bbl / 1e6
        }

    def get_fluid_analysis(self) -> str:
        """Get comprehensive fluid analysis from well data"""
        if not self.fluid_analyzer:
            return "Fluid analysis not available. Well data not loaded."

        context_parts = ["=== FLUID ANALYSIS FROM WELL DATA ===\n"]

        for well_name, df in self.fluid_analyzer.wells.items():
            results = self.fluid_analyzer.analyze_well(df, well_name)

            context_parts.append(f"\nWELL: {well_name}")
            context_parts.append(f"Depth: {results['depth_range'][0]:.0f} - {results['depth_range'][1]:.0f} m")

            if 'summary' in results:
                s = results['summary']
                context_parts.append(f"Oil zones: {s['oil_samples']} samples")
                context_parts.append(f"Gas zones: {s['gas_samples']} samples")
                context_parts.append(f"Water zones: {s['water_samples']} samples")
                context_parts.append(f"Avg Porosity: {s['avg_porosity_pct']:.1f}%")
                context_parts.append(f"Avg Water Sat: {s['avg_sw_pct']:.1f}%")
                context_parts.append(f"Max HC Sat: {s['max_sh_pct']:.1f}%")

            if results.get('pay_summary'):
                p = results['pay_summary']
                context_parts.append(f"PAY ZONE: {p['depth_top_m']:.0f}-{p['depth_base_m']:.0f}m")
                context_parts.append(f"  Porosity: {p['avg_porosity_pct']:.1f}%")
                context_parts.append(f"  HC Sat: {p['avg_sh_pct']:.1f}%")
                context_parts.append(f"  Fluid: {p['dominant_fluid'].upper()}")

            # Detect fluid contacts
            contacts = self._detect_fluid_contacts(df, well_name)
            if contacts:
                context_parts.append("FLUID CONTACTS:")
                for contact in contacts:
                    context_parts.append(f"  {contact['type']}: {contact['depth_m']:.0f}m")

        return "\n".join(context_parts)

    def _detect_fluid_contacts(self, df, well_name: str) -> list:
        """Detect fluid contacts (OWC, GOC, GWC) from well data"""
        contacts = []

        if 'Sw' not in df.columns or 'Sh' not in df.columns:
            return contacts

        # Sort by depth
        df_sorted = df.sort_values('Depth_m')

        # Look for transitions
        prev_fluid = None
        for idx, row in df_sorted.iterrows():
            sw = row['Sw']
            sh = row['Sh']
            depth = row['Depth_m']
            porosity = row.get('Phi_eff', 0)

            # Skip tight zones
            if porosity < 0.05:
                continue

            # Determine current fluid
            if sw > 0.8:
                current_fluid = 'water'
            elif sh > 0.5:
                # Check density for gas vs oil (if available)
                density = row.get('RHOB_gcc', 2.2)
                if density < 2.0:
                    current_fluid = 'gas'
                else:
                    current_fluid = 'oil'
            else:
                current_fluid = 'transition'

            # Detect contact
            if prev_fluid and current_fluid != prev_fluid and current_fluid != 'transition' and prev_fluid != 'transition':
                contact_type = None
                if prev_fluid == 'gas' and current_fluid == 'oil':
                    contact_type = 'GOC (Gas-Oil Contact)'
                elif prev_fluid == 'oil' and current_fluid == 'water':
                    contact_type = 'OWC (Oil-Water Contact)'
                elif prev_fluid == 'gas' and current_fluid == 'water':
                    contact_type = 'GWC (Gas-Water Contact)'
                elif prev_fluid == 'oil' and current_fluid == 'gas':
                    contact_type = 'GOC (Gas-Oil Contact)'
                elif prev_fluid == 'water' and current_fluid == 'oil':
                    contact_type = 'OWC (Oil-Water Contact)'
                elif prev_fluid == 'water' and current_fluid == 'gas':
                    contact_type = 'GWC (Gas-Water Contact)'

                if contact_type:
                    contacts.append({
                        'type': contact_type,
                        'depth_m': depth,
                        'well': well_name
                    })

            prev_fluid = current_fluid

        # Remove duplicates and keep first occurrence
        seen_types = set()
        unique_contacts = []
        for c in contacts:
            if c['type'] not in seen_types:
                seen_types.add(c['type'])
                unique_contacts.append(c)

        return unique_contacts[:3]  # Max 3 contacts per well

    def get_well_fluid_summary(self, well_name: str = None) -> str:
        """Get fluid summary for a specific well or all wells"""
        if not self.fluid_analyzer:
            return "Fluid analysis not available."

        if well_name:
            # Find matching well
            for name, df in self.fluid_analyzer.wells.items():
                if well_name.upper() in name.upper():
                    results = self.fluid_analyzer.analyze_well(df, name)
                    return json.dumps(results, indent=2, default=str)
            return f"Well {well_name} not found."

        # Return summary of all wells
        return self.get_fluid_analysis()


# =============================================================================
# PROCESSING STEPS
# =============================================================================

PROCESSING_STEPS = {
    1: {
        "name": "3D EDA & Quality Assessment",
        "description": "Analyze 3D seismic data quality, geometry, frequency content",
        "folder": "eda",
        "script": "seismic_eda_automation.py",
        "keywords": ["eda", "quality", "qc", "analyze", "assessment", "frequency", "amplitude", "geometry"],
        "output_file": "eda_results.json"
    },
    2: {
        "name": "Dead Trace Repair",
        "description": "Detect and interpolate dead/bad traces in 3D data",
        "folder": "dead_trace",
        "script": "dead_trace_automation.py",
        "keywords": ["dead", "trace", "repair", "fix", "interpolate", "bad", "missing"],
        "output_file": "dead_trace_results.json"
    },
    3: {
        "name": "Well Integration & Petrophysics",
        "description": "Build velocity model, petrophysical analysis, fluid typing",
        "folder": "well_integration",
        "script": "well_integration_automation.py",
        "keywords": ["well", "velocity", "petrophysic", "log", "sonic", "density", "fluid"],
        "output_file": "well_results.json"
    },
    4: {
        "name": "2D Seismic QC",
        "description": "Quality control and inventory of 2D seismic lines",
        "folder": "seismic_2d",
        "script": "seismic_2d_automation.py",
        "keywords": ["2d", "line", "inventory", "qc", "quality"],
        "output_file": "2d_results.json"
    },
    5: {
        "name": "Synthetic Seismograms",
        "description": "Generate synthetic seismograms for well-to-seismic tie",
        "folder": "interpretation",
        "script": "synthetic_seismogram.py",
        "keywords": ["synthetic", "tie", "well", "seismic", "wavelet", "reflectivity"],
        "output_file": "synthetic_outputs/synthetic_results.json"
    },
    6: {
        "name": "Horizon Mapping & Structure",
        "description": "Pick horizons, create time/depth structure maps, identify closures",
        "folder": "interpretation",
        "script": "horizon_mapping.py",
        "keywords": ["horizon", "structure", "map", "closure", "depth", "fault", "pick"],
        "output_file": "structure_outputs/interpretation_results.json"
    },
    7: {
        "name": "Seismic Attributes & DHI",
        "description": "Extract RMS amplitude, sweetness, detect bright spots",
        "folder": "interpretation",
        "script": "seismic_attributes.py",
        "keywords": ["attribute", "amplitude", "rms", "sweetness", "dhi", "bright spot"],
        "output_file": "attribute_outputs/dhi_report.txt"
    },
    8: {
        "name": "Complete Interpretation",
        "description": "Run full interpretation workflow (synthetic + horizons + attributes)",
        "folder": "interpretation",
        "script": "run_interpretation.py",
        "keywords": ["interpretation", "complete", "full", "all"],
        "output_file": "interpretation_master_results.json"
    },
    9: {
        "name": "2D-3D Integration",
        "description": "Integrate 2D lines with 3D volume for regional framework",
        "folder": "integration_2d3d",
        "script": "integration_2d3d_automation.py",
        "keywords": ["2d", "3d", "integration", "regional", "composite", "tie"],
        "output_file": "integration_results.json"
    },
    10: {
        "name": "REAL Interpretation (PhD Critical)",
        "description": "Run complete interpretation from ACTUAL SEGY data: well ties, horizons, attributes, faults, volumetrics",
        "folder": "interpretation",
        "script": "real_interpretation.py",
        "keywords": ["real", "actual", "segy", "complete", "phd", "thesis"],
        "output_file": "real_outputs/interpretation_results.json"
    }
}

# Interpretation outputs for display
INTERPRETATION_OUTPUTS = {
    "synthetic": {
        "name": "Synthetic Seismograms",
        "folder": "interpretation/synthetic_outputs",
        "files": ["synthetic_BULTE-1.png", "synthetic_HERWA-01.png", "synthetic_MASU-1.png", "synthetic_KASADE-01.png"],
        "report": "synthetic_report.txt"
    },
    "structure": {
        "name": "Structure Maps",
        "folder": "interpretation/structure_outputs",
        "files": ["time_structure_Top_Chad_Fm.png", "time_structure_Top_Fika_Shale.png",
                 "time_structure_Top_Gongila_Fm.png", "time_structure_Top_Bima_Sst.png",
                 "depth_structure_Top_Bima_Sst.png", "prospect_map_Bima.png"],
        "report": "structure_report.txt"
    },
    "attributes": {
        "name": "Seismic Attributes",
        "folder": "interpretation/attribute_outputs",
        "files": ["rms_amplitude_map.png", "sweetness_map.png"],
        "report": "dhi_report.txt"
    }
}


# =============================================================================
# AI INTERPRETER
# =============================================================================

class SeismicAIInterpreter:
    """AI-powered seismic interpretation assistant"""

    SYSTEM_PROMPT = """You are an expert seismic interpreter and petroleum geoscientist for the Bornu Chad Basin, Nigeria.

## CAPABILITIES:
1. Interpret seismic processing results
2. Recommend drilling locations with specific UTM coordinates
3. Calculate oil/gas saturation, porosity, STOIIP, GIIP
4. Identify fluid types (oil, gas, water) and contacts (OWC, GOC, GWC)
5. Interpret seismic images (sections, maps, attributes)
6. Generate technical reports

## BASIN STRATIGRAPHY:
| Formation | Age | Depth (m) | Significance |
|-----------|-----|-----------|--------------|
| Chad Fm | Quaternary | 0-1000 | Aquifer |
| Fika Shale | Turonian | 1000-2500 | SOURCE ROCK |
| Gongila Fm | Cenomanian | 2500-3500 | Secondary reservoir |
| Bima Fm | Albian | 3500-5000 | PRIMARY RESERVOIR |

## COORDINATE SYSTEM:
- UTM Zone 33N (Nigeria)
- X Range: 325,000 - 377,000 m
- Y Range: 1,420,000 - 1,460,000 m

## FLUID IDENTIFICATION:
- Oil zones: Sw < 50%, high resistivity, normal density
- Gas zones: Sw < 50%, very high resistivity, low density, N-D crossover
- Water zones: Sw > 80%, low resistivity
- GOC (Gas-Oil Contact): Transition from gas to oil with depth
- OWC (Oil-Water Contact): Transition from oil to water with depth
- GWC (Gas-Water Contact): Direct gas to water transition

## RESERVOIR PROPERTIES FROM WELLS:
- Average Pay Porosity: 18-23%
- Average HC Saturation in Pay: 62-67%
- Dominant Fluid: Mixed OIL + GAS system
- Best Pay Zones: 455-1518m depth

## VOLUMETRICS:
- STOIIP = GRV × NTG × Porosity × (1-Sw) / Bo
- GIIP = GRV × NTG × Porosity × (1-Sw) / Bg
- EUR = STOIIP × Recovery Factor (typically 35% for oil, 75% for gas)

Always provide SPECIFIC coordinates and QUANTITATIVE answers with units."""

    def __init__(self, ollama: OllamaClient, kb: SeismicKnowledgeBase, state: StateManager):
        self.ollama = ollama
        self.kb = kb
        self.state = state

    def interpret_query(self, query: str) -> str:
        context = self.kb.load_all_results()
        query_lower = query.lower()

        # Check for examiner-type questions first
        examiner_answer = self._check_examiner_questions(query_lower)
        if examiner_answer:
            context += f"\n\n=== PREPARED ANSWER ===\n{examiner_answer}"

        # Drilling questions
        if any(word in query_lower for word in ["drill", "location", "prospect", "where to drill", "recommend"]):
            context = self.kb.get_drilling_context()
            context += "\n\n" + self.kb.get_fluid_analysis()

        # Methodology questions (examiner favorites)
        elif any(word in query_lower for word in ["how did you", "methodology", "method", "approach", "why did you"]):
            context = self._get_methodology_context()

        # Synthetic/well tie questions
        elif any(word in query_lower for word in ["synthetic", "well tie", "seismogram", "wavelet", "reflectivity"]):
            context = self._get_interpretation_context("synthetic")

        # Horizon/structure questions
        elif any(word in query_lower for word in ["horizon", "structure", "map", "closure", "fault", "pick"]):
            context = self._get_interpretation_context("structure")

        # Velocity model questions
        elif any(word in query_lower for word in ["velocity", "depth conversion", "v(z)", "time to depth"]):
            context = self._get_methodology_context()

        # Resolution questions
        elif any(word in query_lower for word in ["resolution", "frequency", "bandwidth", "tuning"]):
            context = self._get_methodology_context()

        # AVO questions
        elif any(word in query_lower for word in ["avo", "amplitude variation", "offset", "pre-stack"]):
            context += "\n\nIMPORTANT: AVO analysis requires pre-stack data. This dataset is POST-STACK (already stacked), so AVO analysis is NOT possible. Post-stack amplitude attributes (RMS, sweetness) were used instead for DHI detection."

        # DHI questions
        elif any(word in query_lower for word in ["dhi", "bright spot", "flat spot", "dim spot", "hydrocarbon indicator"]):
            context = self._get_interpretation_context("attributes")

        # Fluid/saturation questions
        elif any(word in query_lower for word in ["oil", "gas", "water", "fluid", "saturation", "sw", "so", "sg", "hydrocarbon"]):
            context = self.kb.get_fluid_analysis()

        # Contact questions (OWC, GOC, GWC)
        elif any(word in query_lower for word in ["contact", "owc", "goc", "gwc", "oil-water", "gas-oil", "gas-water"]):
            context = self.kb.get_fluid_analysis()
            context += "\n\nFLUID CONTACT DEFINITIONS:"
            context += "\n- OWC (Oil-Water Contact): Mean depth 1546m (range 1064-2028m)"
            context += "\n- GOC (Gas-Oil Contact): Mean depth 1006m (range 405-3119m)"
            context += "\n- GWC (Gas-Water Contact): ~1050m depth"

        # Volumetric questions
        elif any(word in query_lower for word in ["stoiip", "giip", "volume", "reserves", "eur", "mmstb", "bcf", "p10", "p50", "p90", "monte carlo"]):
            context = self.kb.get_fluid_analysis()
            context += "\n\nMONTE CARLO VOLUMETRICS (10,000 simulations):"
            context += "\n- P10 (Low case): 238 MMSTB"
            context += "\n- P50 (Best estimate): 337 MMSTB"
            context += "\n- P90 (High case): 463 MMSTB"
            context += "\n- Recovery Factor: 35%"
            context += "\n- EUR (P50): 118 MMSTB"

        # Well-specific questions
        elif any(well in query_lower for well in ["bulte", "herwa", "masu", "kasade", "ngamma", "ngor", "gaibu", "well"]):
            context = self.kb.get_fluid_analysis()
            for well_name in ["BULTE-1", "HERWA-01", "MASU-1", "KASADE-01", "NGAMMAEAST-1", "NGORNORTH-1"]:
                if well_name.lower().replace("-", "").replace("1", "").replace("01", "") in query_lower.replace("-", "").replace("1", "").replace("01", ""):
                    well_data = self.kb.get_well_fluid_summary(well_name)
                    context += f"\n\nDETAILED DATA FOR {well_name}:\n{well_data}"
                    break

        # Risk questions
        elif any(word in query_lower for word in ["risk", "uncertainty", "chance", "probability", "confidence"]):
            context += "\n\nRISK ASSESSMENT:"
            context += "\n- TRAP RISK (Low): 4-way closure confirmed"
            context += "\n- RESERVOIR RISK (Low-Moderate): 18-23% porosity from wells"
            context += "\n- SOURCE RISK (Low): Fika Shale proven source"
            context += "\n- SEAL RISK (Moderate): Need to confirm seal thickness"
            context += "\n- CHARGE RISK (Moderate): Migration pathway uncertain"
            context += "\n- Overall Geological Probability of Success: ~45%"

        return self.ollama.chat(query, self.SYSTEM_PROMPT, context)

    def _check_examiner_questions(self, query_lower: str) -> str:
        """Check if the query matches known examiner questions and return prepared answer"""
        if not self.kb.geological_knowledge:
            return ""

        examiner_qs = self.kb.geological_knowledge.get("examiner_questions", {})

        # Keywords to category mapping
        keyword_map = {
            "methodology": ["how did you tie", "well tie", "pick horizon", "velocity model", "attribute", "fluid contact"],
            "data_quality": ["data quality", "resolution", "dead trace", "snr", "frequency"],
            "structure": ["structural style", "closure", "fault", "trap"],
            "reservoir": ["reservoir", "porosity", "saturation", "archie", "source rock"],
            "volumetrics": ["stoiip", "calculate", "p10", "p50", "p90", "uncertainty", "monte carlo"],
            "dhi_analysis": ["dhi", "bright spot", "avo", "confidence"],
            "recommendations": ["recommend", "drilling", "risk", "next step"]
        }

        for category, keywords in keyword_map.items():
            for keyword in keywords:
                if keyword in query_lower:
                    cat_questions = examiner_qs.get(category, {})
                    for q_id, q_data in cat_questions.items():
                        q_text = q_data.get("question", "").lower()
                        # Check if query is similar to this question
                        if any(word in query_lower for word in q_text.split()[:5]):
                            return q_data.get("answer", "")

        return ""

    def _get_methodology_context(self) -> str:
        """Get methodology context from knowledge base"""
        context = "=== METHODOLOGY INFORMATION ===\n"

        if self.kb.geological_knowledge:
            examiner_qs = self.kb.geological_knowledge.get("examiner_questions", {})
            methodology = examiner_qs.get("methodology", {})
            for q_id, q_data in methodology.items():
                context += f"\nQ: {q_data.get('question', '')}"
                context += f"\nA: {q_data.get('answer', '')}\n"

        return context

    def _get_interpretation_context(self, interp_type: str) -> str:
        """Get interpretation results context"""
        context = ""

        # Try to load interpretation results
        interp_dir = BASE_DIR / "interpretation"

        if interp_type == "synthetic":
            results_file = interp_dir / "synthetic_outputs" / "synthetic_results.json"
            report_file = interp_dir / "synthetic_outputs" / "synthetic_report.txt"
        elif interp_type == "structure":
            results_file = interp_dir / "structure_outputs" / "interpretation_results.json"
            report_file = interp_dir / "structure_outputs" / "structure_report.txt"
        elif interp_type == "attributes":
            results_file = None
            report_file = interp_dir / "attribute_outputs" / "dhi_report.txt"
        else:
            return context

        if results_file and results_file.exists():
            try:
                with open(results_file) as f:
                    data = json.load(f)
                context += f"\n=== {interp_type.upper()} RESULTS ===\n"
                context += json.dumps(data, indent=2)[:2000]
            except:
                pass

        if report_file and report_file.exists():
            try:
                with open(report_file) as f:
                    context += f"\n\n=== {interp_type.upper()} REPORT ===\n"
                    context += f.read()[:2000]
            except:
                pass

        return context

    def interpret_image(self, image_path: str, additional_prompt: str = "") -> str:
        base_prompt = """Analyze this seismic image. Identify:
1. Key reflectors and their characteristics
2. Structural features (faults, folds, closures)
3. Potential hydrocarbon indicators (bright spots, flat spots, dim spots)
4. Data quality assessment
5. Geological interpretation

Provide specific observations with depths/times where visible."""

        if additional_prompt:
            base_prompt += f"\n\nAdditional focus: {additional_prompt}"

        return self.ollama.interpret_image(image_path, base_prompt)

    def generate_report(self, report_type: str = "summary") -> str:
        context = self.kb.load_all_results()

        if report_type == "summary":
            prompt = """Generate a technical summary report with:
1. Executive Summary
2. Data Quality Assessment
3. Key Findings
4. Recommendations
5. Next Steps

Use the processing results provided. Be specific and quantitative."""

        elif report_type == "drilling":
            prompt = """Generate a drilling recommendation report with:
1. Prospect Summary
2. Recommended Drilling Locations (with coordinates)
3. Target Formations and Depths
4. Risk Assessment
5. Volumetric Estimates
6. Operational Recommendations"""

        elif report_type == "full":
            prompt = """Generate a comprehensive technical report suitable for a PhD thesis chapter:
1. Introduction
2. Data and Methods
3. Results
   - Seismic Quality Assessment
   - Well Integration
   - Structural Interpretation
   - Reservoir Characterization
4. Discussion
5. Conclusions
6. Recommendations"""

        else:
            prompt = f"Generate a {report_type} report based on the seismic processing results."

        return self.ollama.chat(prompt, self.SYSTEM_PROMPT, context)


# =============================================================================
# GUI APPLICATION
# =============================================================================

class SeismicAIAssistantGUI:
    """Enhanced GUI Application"""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Seismic AI Assistant v2.0 - PhD Research")
        self.root.geometry("1600x950")
        self.root.minsize(1400, 800)

        # Initialize components
        self.ollama = OllamaClient()
        self.state = StateManager()
        self.kb = SeismicKnowledgeBase()
        self.interpreter = SeismicAIInterpreter(self.ollama, self.kb, self.state)

        # Processing queue for live output
        self.process_queue = queue.Queue()
        self.current_process = None

        # Colors
        self.colors = {
            'bg': '#1a1b26',
            'fg': '#c0caf5',
            'accent': '#7aa2f7',
            'success': '#9ece6a',
            'warning': '#e0af68',
            'error': '#f7768e',
            'surface': '#24283b',
            'overlay': '#414868',
            'terminal_bg': '#1a1b26',
            'terminal_fg': '#a9b1d6'
        }

        self.root.configure(bg=self.colors['bg'])

        # Build UI
        self._create_menu()
        self._create_notebook()
        self._create_status_bar()

        # Check connection
        self.check_ollama_connection()

        # Welcome message
        self._show_welcome()

        # Start queue processor
        self._process_queue()

    def _create_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New Project", command=self.new_project)
        file_menu.add_command(label="Open Project", command=self.open_project)
        file_menu.add_separator()
        file_menu.add_command(label="Export Report", command=self.export_report)
        file_menu.add_command(label="Export Chat", command=self.export_chat)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        # Processing menu
        proc_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Processing", menu=proc_menu)
        for step_num, step_info in PROCESSING_STEPS.items():
            proc_menu.add_command(
                label=f"Step {step_num}: {step_info['name']}",
                command=lambda n=step_num: self.run_processing_step(n)
            )
        proc_menu.add_separator()
        proc_menu.add_command(label="Run All Pending", command=self.run_all_steps)
        proc_menu.add_command(label="Stop Current", command=self.stop_processing)

        # Reports menu
        report_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Reports", menu=report_menu)
        report_menu.add_command(label="Summary Report", command=lambda: self.generate_report("summary"))
        report_menu.add_command(label="Drilling Report", command=lambda: self.generate_report("drilling"))
        report_menu.add_command(label="Full Technical Report", command=lambda: self.generate_report("full"))

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)

    def _create_notebook(self):
        """Create tabbed interface"""
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Tab 1: Chat & Interpretation
        self.chat_tab = tk.Frame(self.notebook, bg=self.colors['bg'])
        self.notebook.add(self.chat_tab, text="  Chat & Interpretation  ")
        self._create_chat_tab()

        # Tab 2: Data Configuration
        self.data_tab = tk.Frame(self.notebook, bg=self.colors['bg'])
        self.notebook.add(self.data_tab, text="  Data Configuration  ")
        self._create_data_tab()

        # Tab 3: Processing & Progress
        self.process_tab = tk.Frame(self.notebook, bg=self.colors['bg'])
        self.notebook.add(self.process_tab, text="  Processing  ")
        self._create_processing_tab()

        # Tab 4: Interpretation Results
        self.interp_tab = tk.Frame(self.notebook, bg=self.colors['bg'])
        self.notebook.add(self.interp_tab, text="  Interpretation  ")
        self._create_interpretation_tab()

        # Tab 5: Well Logs & Formation Picks
        self.welllogs_tab = tk.Frame(self.notebook, bg=self.colors['bg'])
        self.notebook.add(self.welllogs_tab, text="  Well Logs & Picks  ")
        self._create_welllogs_tab()

        # Tab 6: Image Interpretation
        self.image_tab = tk.Frame(self.notebook, bg=self.colors['bg'])
        self.notebook.add(self.image_tab, text="  Image Analysis  ")
        self._create_image_tab()

        # Tab 7: Interactive Seismic Viewer
        self.seismic_viewer_tab = tk.Frame(self.notebook, bg=self.colors['bg'])
        self.notebook.add(self.seismic_viewer_tab, text="  Seismic Viewer  ")
        self._create_seismic_viewer_tab()

        # Tab 8: Well-to-Seismic Tie Validation
        self.well_tie_tab = tk.Frame(self.notebook, bg=self.colors['bg'])
        self.notebook.add(self.well_tie_tab, text="  Well Tie Validation  ")
        self._create_well_tie_tab()

    def _create_chat_tab(self):
        """Create chat interface tab"""
        # Main container
        main_frame = tk.Frame(self.chat_tab, bg=self.colors['bg'])
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Left: Chat area
        chat_frame = tk.Frame(main_frame, bg=self.colors['bg'])
        chat_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        # Chat display
        self.chat_display = scrolledtext.ScrolledText(
            chat_frame,
            wrap=tk.WORD,
            font=('Consolas', 11),
            bg=self.colors['surface'],
            fg=self.colors['fg'],
            insertbackground=self.colors['fg'],
            relief=tk.FLAT,
            padx=15,
            pady=15
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True)
        self.chat_display.config(state=tk.DISABLED)

        # Configure tags
        self.chat_display.tag_configure('user', foreground=self.colors['accent'], font=('Consolas', 11, 'bold'))
        self.chat_display.tag_configure('ai', foreground=self.colors['success'])
        self.chat_display.tag_configure('system', foreground=self.colors['warning'])
        self.chat_display.tag_configure('error', foreground=self.colors['error'])

        # Input area
        input_frame = tk.Frame(chat_frame, bg=self.colors['bg'])
        input_frame.pack(fill=tk.X, pady=(5, 0))

        self.chat_input = tk.Text(
            input_frame,
            height=3,
            font=('Consolas', 11),
            bg=self.colors['surface'],
            fg=self.colors['fg'],
            insertbackground=self.colors['fg'],
            relief=tk.FLAT,
            padx=10,
            pady=10
        )
        self.chat_input.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        self.chat_input.bind('<Return>', self.on_enter)
        self.chat_input.bind('<Shift-Return>', lambda e: None)

        # Buttons
        btn_frame = tk.Frame(input_frame, bg=self.colors['bg'])
        btn_frame.pack(side=tk.RIGHT, fill=tk.Y)

        self.send_btn = tk.Button(
            btn_frame,
            text="Send",
            font=('Segoe UI', 11, 'bold'),
            bg=self.colors['accent'],
            fg='white',
            relief=tk.FLAT,
            width=12,
            command=self.send_message
        )
        self.send_btn.pack(fill=tk.X, pady=(0, 3))

        # Quick actions
        quick_frame = tk.Frame(btn_frame, bg=self.colors['bg'])
        quick_frame.pack(fill=tk.X)

        for text, query in [
            ("Drill Location", "What is the best drilling location? Provide UTM coordinates."),
            ("STOIIP", "Calculate STOIIP for a 15 km2 prospect with 60m net pay."),
            ("Summary", "Give me a summary of all completed processing steps.")
        ]:
            tk.Button(
                quick_frame,
                text=text,
                font=('Segoe UI', 9),
                bg=self.colors['overlay'],
                fg=self.colors['fg'],
                relief=tk.FLAT,
                command=lambda q=query: self.quick_query(q)
            ).pack(fill=tk.X, pady=1)

        # Right: Model selection and status
        sidebar = tk.Frame(main_frame, bg=self.colors['surface'], width=280)
        sidebar.pack(side=tk.RIGHT, fill=tk.Y)
        sidebar.pack_propagate(False)

        # Model selection
        tk.Label(
            sidebar,
            text="AI Model",
            font=('Segoe UI', 12, 'bold'),
            bg=self.colors['surface'],
            fg=self.colors['accent']
        ).pack(pady=(15, 5), padx=10, anchor=tk.W)

        self.model_var = tk.StringVar(value="qwen3:32b")
        self.model_combo = ttk.Combobox(
            sidebar,
            textvariable=self.model_var,
            state="readonly",
            width=30
        )
        self.model_combo.pack(padx=10, fill=tk.X)
        self.model_combo.bind("<<ComboboxSelected>>", self.on_model_change)

        self.connection_label = tk.Label(
            sidebar,
            text="Checking...",
            font=('Segoe UI', 9),
            bg=self.colors['surface'],
            fg=self.colors['warning']
        )
        self.connection_label.pack(padx=10, anchor=tk.W, pady=5)

        ttk.Separator(sidebar, orient='horizontal').pack(fill=tk.X, padx=10, pady=10)

        # Processing status
        tk.Label(
            sidebar,
            text="Processing Status",
            font=('Segoe UI', 12, 'bold'),
            bg=self.colors['surface'],
            fg=self.colors['accent']
        ).pack(pady=(5, 10), padx=10, anchor=tk.W)

        self.status_labels = {}
        for step_num, step_info in PROCESSING_STEPS.items():
            frame = tk.Frame(sidebar, bg=self.colors['surface'])
            frame.pack(fill=tk.X, padx=10, pady=2)

            status = "Done" if self.state.is_completed(step_num) else "Pending"
            color = self.colors['success'] if status == "Done" else self.colors['overlay']

            tk.Label(
                frame,
                text=f"{step_num}.",
                font=('Segoe UI', 9),
                bg=self.colors['surface'],
                fg=self.colors['fg'],
                width=2
            ).pack(side=tk.LEFT)

            tk.Label(
                frame,
                text=step_info['name'][:25],
                font=('Segoe UI', 9),
                bg=self.colors['surface'],
                fg=self.colors['fg']
            ).pack(side=tk.LEFT, fill=tk.X, expand=True)

            status_lbl = tk.Label(
                frame,
                text=status,
                font=('Segoe UI', 8),
                bg=color,
                fg='white' if status == "Done" else self.colors['fg'],
                padx=5
            )
            status_lbl.pack(side=tk.RIGHT)
            self.status_labels[step_num] = status_lbl

    def _create_data_tab(self):
        """Create data configuration tab"""
        # Main container with padding
        main_frame = tk.Frame(self.data_tab, bg=self.colors['bg'], padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        tk.Label(
            main_frame,
            text="Data Configuration",
            font=('Segoe UI', 18, 'bold'),
            bg=self.colors['bg'],
            fg=self.colors['accent']
        ).pack(anchor=tk.W, pady=(0, 20))

        # Create scrollable frame
        canvas = tk.Canvas(main_frame, bg=self.colors['bg'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=self.colors['bg'])

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Data input fields
        self.data_entries = {}

        data_fields = [
            ("3D Seismic Volume (SEGY)", "segy_3d", "file", [("SEGY files", "*.segy *.sgy *.SEGY *.SGY")]),
            ("2D Seismic Folder", "segy_2d_folder", "folder", None),
            ("Well Data Folder (LAS files)", "well_folder", "folder", None),
            ("Well Header File (Excel/CSV)", "well_header", "file", [("Excel/CSV", "*.xlsx *.xls *.csv")]),
            ("Output Folder", "output_folder", "folder", None),
        ]

        for label_text, key, input_type, filetypes in data_fields:
            frame = tk.Frame(scrollable_frame, bg=self.colors['surface'], padx=15, pady=15)
            frame.pack(fill=tk.X, pady=5)

            tk.Label(
                frame,
                text=label_text,
                font=('Segoe UI', 11, 'bold'),
                bg=self.colors['surface'],
                fg=self.colors['fg']
            ).pack(anchor=tk.W)

            entry_frame = tk.Frame(frame, bg=self.colors['surface'])
            entry_frame.pack(fill=tk.X, pady=(5, 0))

            entry = tk.Entry(
                entry_frame,
                font=('Consolas', 10),
                bg=self.colors['overlay'],
                fg=self.colors['fg'],
                insertbackground=self.colors['fg'],
                relief=tk.FLAT
            )
            entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5), ipady=8)

            # Load saved path
            saved_path = self.state.get_data_path(key)
            if saved_path:
                entry.insert(0, saved_path)

            self.data_entries[key] = entry

            if input_type == "file":
                btn = tk.Button(
                    entry_frame,
                    text="Browse File",
                    font=('Segoe UI', 9),
                    bg=self.colors['accent'],
                    fg='white',
                    relief=tk.FLAT,
                    command=lambda e=entry, ft=filetypes: self.browse_file(e, ft)
                )
            else:
                btn = tk.Button(
                    entry_frame,
                    text="Browse Folder",
                    font=('Segoe UI', 9),
                    bg=self.colors['accent'],
                    fg='white',
                    relief=tk.FLAT,
                    command=lambda e=entry: self.browse_folder(e)
                )
            btn.pack(side=tk.RIGHT)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Save button
        btn_frame = tk.Frame(main_frame, bg=self.colors['bg'])
        btn_frame.pack(fill=tk.X, pady=20)

        tk.Button(
            btn_frame,
            text="Save Configuration",
            font=('Segoe UI', 12, 'bold'),
            bg=self.colors['success'],
            fg='white',
            relief=tk.FLAT,
            padx=30,
            pady=10,
            command=self.save_data_config
        ).pack(side=tk.LEFT)

        tk.Button(
            btn_frame,
            text="Validate Data",
            font=('Segoe UI', 12),
            bg=self.colors['accent'],
            fg='white',
            relief=tk.FLAT,
            padx=30,
            pady=10,
            command=self.validate_data
        ).pack(side=tk.LEFT, padx=10)

    def _create_processing_tab(self):
        """Create processing tab with live output"""
        # Split view
        paned = tk.PanedWindow(self.process_tab, orient=tk.HORIZONTAL, bg=self.colors['bg'])
        paned.pack(fill=tk.BOTH, expand=True)

        # Left: Step selection with scrollable frame
        left_frame = tk.Frame(paned, bg=self.colors['surface'], width=350)
        paned.add(left_frame)

        tk.Label(
            left_frame,
            text="Processing Steps",
            font=('Segoe UI', 14, 'bold'),
            bg=self.colors['surface'],
            fg=self.colors['accent']
        ).pack(pady=15, padx=15, anchor=tk.W)

        # Create scrollable container for steps
        steps_container = tk.Frame(left_frame, bg=self.colors['surface'])
        steps_container.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(steps_container, bg=self.colors['surface'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(steps_container, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=self.colors['surface'])

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Make scrollable frame expand to canvas width
        def _configure_canvas(event):
            canvas.itemconfig(canvas_window, width=event.width)
        canvas.bind("<Configure>", _configure_canvas)

        # Enable mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Steps list
        self.step_frames = {}
        for step_num, step_info in PROCESSING_STEPS.items():
            frame = tk.Frame(scrollable_frame, bg=self.colors['overlay'], padx=10, pady=10)
            frame.pack(fill=tk.X, padx=10, pady=3)

            # Header
            header = tk.Frame(frame, bg=self.colors['overlay'])
            header.pack(fill=tk.X)

            tk.Label(
                header,
                text=f"Step {step_num}",
                font=('Segoe UI', 10, 'bold'),
                bg=self.colors['overlay'],
                fg=self.colors['accent']
            ).pack(side=tk.LEFT)

            status = "Done" if self.state.is_completed(step_num) else "Pending"
            status_color = self.colors['success'] if status == "Done" else self.colors['warning']

            proc_status = tk.Label(
                header,
                text=status,
                font=('Segoe UI', 8),
                bg=status_color,
                fg='black',
                padx=5
            )
            proc_status.pack(side=tk.RIGHT)

            tk.Label(
                frame,
                text=step_info['name'],
                font=('Segoe UI', 10),
                bg=self.colors['overlay'],
                fg=self.colors['fg']
            ).pack(anchor=tk.W, pady=(5, 0))

            tk.Label(
                frame,
                text=step_info['description'],
                font=('Segoe UI', 9),
                bg=self.colors['overlay'],
                fg=self.colors['fg'],
                wraplength=300
            ).pack(anchor=tk.W)

            # Buttons
            btn_frame = tk.Frame(frame, bg=self.colors['overlay'])
            btn_frame.pack(fill=tk.X, pady=(10, 0))

            run_btn = tk.Button(
                btn_frame,
                text="Run",
                font=('Segoe UI', 9),
                bg=self.colors['accent'],
                fg='white',
                relief=tk.FLAT,
                padx=15,
                command=lambda n=step_num: self.run_processing_step(n)
            )
            run_btn.pack(side=tk.LEFT)

            self.step_frames[step_num] = {
                'frame': frame,
                'status': proc_status,
                'run_btn': run_btn
            }

        # Right: Terminal output
        right_frame = tk.Frame(paned, bg=self.colors['bg'])
        paned.add(right_frame)

        # Terminal header
        term_header = tk.Frame(right_frame, bg=self.colors['surface'])
        term_header.pack(fill=tk.X)

        tk.Label(
            term_header,
            text="Processing Output",
            font=('Segoe UI', 12, 'bold'),
            bg=self.colors['surface'],
            fg=self.colors['accent']
        ).pack(side=tk.LEFT, padx=15, pady=10)

        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            term_header,
            variable=self.progress_var,
            maximum=100,
            length=200
        )
        self.progress_bar.pack(side=tk.RIGHT, padx=15, pady=10)

        self.progress_label = tk.Label(
            term_header,
            text="Ready",
            font=('Segoe UI', 10),
            bg=self.colors['surface'],
            fg=self.colors['fg']
        )
        self.progress_label.pack(side=tk.RIGHT, padx=10)

        # Terminal output
        self.terminal = scrolledtext.ScrolledText(
            right_frame,
            wrap=tk.WORD,
            font=('Consolas', 10),
            bg=self.colors['terminal_bg'],
            fg=self.colors['terminal_fg'],
            insertbackground=self.colors['terminal_fg'],
            relief=tk.FLAT,
            padx=15,
            pady=15
        )
        self.terminal.pack(fill=tk.BOTH, expand=True)
        self.terminal.config(state=tk.DISABLED)

        # Terminal tags
        self.terminal.tag_configure('info', foreground=self.colors['accent'])
        self.terminal.tag_configure('success', foreground=self.colors['success'])
        self.terminal.tag_configure('warning', foreground=self.colors['warning'])
        self.terminal.tag_configure('error', foreground=self.colors['error'])

        # Control buttons
        ctrl_frame = tk.Frame(right_frame, bg=self.colors['bg'])
        ctrl_frame.pack(fill=tk.X, pady=10, padx=10)

        tk.Button(
            ctrl_frame,
            text="Clear Output",
            font=('Segoe UI', 10),
            bg=self.colors['overlay'],
            fg=self.colors['fg'],
            relief=tk.FLAT,
            command=self.clear_terminal
        ).pack(side=tk.LEFT)

        tk.Button(
            ctrl_frame,
            text="Stop Processing",
            font=('Segoe UI', 10),
            bg=self.colors['error'],
            fg='white',
            relief=tk.FLAT,
            command=self.stop_processing
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            ctrl_frame,
            text="Run All Pending",
            font=('Segoe UI', 10),
            bg=self.colors['success'],
            fg='white',
            relief=tk.FLAT,
            command=self.run_all_steps
        ).pack(side=tk.RIGHT)

    def _create_interpretation_tab(self):
        """Create interpretation results tab with maps and reports"""
        # Main container
        main_frame = tk.Frame(self.interp_tab, bg=self.colors['bg'], padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        tk.Label(
            main_frame,
            text="Interpretation Results & Maps",
            font=('Segoe UI', 18, 'bold'),
            bg=self.colors['bg'],
            fg=self.colors['accent']
        ).pack(anchor=tk.W)

        tk.Label(
            main_frame,
            text="View generated structure maps, synthetics, and interpretation reports",
            font=('Segoe UI', 10),
            bg=self.colors['bg'],
            fg=self.colors['fg']
        ).pack(anchor=tk.W, pady=(5, 20))

        # Split view
        content = tk.Frame(main_frame, bg=self.colors['bg'])
        content.pack(fill=tk.BOTH, expand=True)

        # Left: Quick Actions
        left_frame = tk.Frame(content, bg=self.colors['surface'], width=300)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_frame.pack_propagate(False)

        tk.Label(
            left_frame,
            text="Quick Actions",
            font=('Segoe UI', 12, 'bold'),
            bg=self.colors['surface'],
            fg=self.colors['fg']
        ).pack(anchor=tk.W, padx=15, pady=(15, 10))

        # Interpretation workflow buttons
        actions = [
            ("View 3D Seismic + Wells", self.generate_seismic_with_wells),
            ("Generate Survey Basemap", self.generate_survey_basemap),
            ("Run Synthetic Seismograms", self.run_synthetic_workflow),
            ("Run Horizon Mapping", self.run_horizon_workflow),
            ("Run Seismic Attributes", self.run_attribute_workflow),
            ("Run Complete Interpretation", self.run_full_interpretation),
            ("Open Output Folder", self.open_interpretation_folder),
        ]

        for text, cmd in actions:
            btn = tk.Button(
                left_frame,
                text=text,
                font=('Segoe UI', 10),
                bg=self.colors['overlay'],
                fg=self.colors['fg'],
                relief=tk.FLAT,
                anchor=tk.W,
                padx=10,
                command=cmd
            )
            btn.pack(fill=tk.X, padx=15, pady=3)

        # Separator
        tk.Frame(left_frame, bg=self.colors['overlay'], height=1).pack(fill=tk.X, padx=15, pady=15)

        # Generated Maps List
        tk.Label(
            left_frame,
            text="Generated Maps",
            font=('Segoe UI', 12, 'bold'),
            bg=self.colors['surface'],
            fg=self.colors['fg']
        ).pack(anchor=tk.W, padx=15, pady=(0, 10))

        # Scrollable list of maps
        self.maps_listbox = tk.Listbox(
            left_frame,
            font=('Segoe UI', 9),
            bg=self.colors['overlay'],
            fg=self.colors['fg'],
            selectbackground=self.colors['accent'],
            relief=tk.FLAT,
            height=15
        )
        self.maps_listbox.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 10))
        self.maps_listbox.bind('<<ListboxSelect>>', self.on_map_select)

        tk.Button(
            left_frame,
            text="Refresh Map List",
            font=('Segoe UI', 9),
            bg=self.colors['overlay'],
            fg=self.colors['fg'],
            relief=tk.FLAT,
            command=self.refresh_maps_list
        ).pack(fill=tk.X, padx=15, pady=(0, 15))

        # Right: Map Preview and Reports
        right_frame = tk.Frame(content, bg=self.colors['bg'])
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Map preview area
        preview_frame = tk.Frame(right_frame, bg=self.colors['surface'])
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        tk.Label(
            preview_frame,
            text="Map Preview",
            font=('Segoe UI', 12, 'bold'),
            bg=self.colors['surface'],
            fg=self.colors['fg']
        ).pack(anchor=tk.W, padx=15, pady=(15, 10))

        self.map_preview_label = tk.Label(
            preview_frame,
            text="Select a map from the list to preview\n\nGenerated maps include:\n• Time-structure maps\n• Depth-structure maps\n• Synthetic seismograms\n• Attribute maps (RMS, Sweetness)\n• DHI analysis maps",
            font=('Segoe UI', 11),
            bg=self.colors['overlay'],
            fg=self.colors['fg'],
            justify=tk.CENTER
        )
        self.map_preview_label.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))

        # Interpretation Summary
        summary_frame = tk.Frame(right_frame, bg=self.colors['surface'], height=150)
        summary_frame.pack(fill=tk.X)
        summary_frame.pack_propagate(False)

        tk.Label(
            summary_frame,
            text="Interpretation Summary",
            font=('Segoe UI', 12, 'bold'),
            bg=self.colors['surface'],
            fg=self.colors['fg']
        ).pack(anchor=tk.W, padx=15, pady=(15, 10))

        self.interp_summary = tk.Text(
            summary_frame,
            font=('Consolas', 9),
            bg=self.colors['overlay'],
            fg=self.colors['fg'],
            relief=tk.FLAT,
            height=5,
            wrap=tk.WORD
        )
        self.interp_summary.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))
        self.interp_summary.insert(tk.END, "Run interpretation workflow to generate summary...")
        self.interp_summary.config(state=tk.DISABLED)

        # Load existing maps on startup
        self.root.after(1000, self.refresh_maps_list)

    def generate_seismic_with_wells(self):
        """Generate a multi-panel view showing actual seismic data with well locations"""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
        from matplotlib.lines import Line2D
        import numpy as np

        try:
            # Check for segyio
            try:
                import segyio
            except ImportError:
                messagebox.showerror("Error", "segyio not installed. Run: pip install segyio")
                return

            # Find seismic file
            segy_path = BASE_DIR / 'outputs' / '3D Bornu Chad_cleaned.segy'
            if not segy_path.exists():
                segy_path = BASE_DIR / 'seismic_3d' / '3D Bornu Chad.segy'
            if not segy_path.exists():
                # Ask user to select
                segy_path = filedialog.askopenfilename(
                    title="Select 3D SEGY File",
                    initialdir=str(BASE_DIR),
                    filetypes=[("SEGY files", "*.segy *.sgy"), ("All files", "*")]
                )
                if not segy_path:
                    return
                segy_path = Path(segy_path)

            self.interp_summary.config(state=tk.NORMAL)
            self.interp_summary.delete(1.0, tk.END)
            self.interp_summary.insert(tk.END, f"Loading seismic data from:\n{segy_path}\n\nThis may take a moment...")
            self.interp_summary.config(state=tk.DISABLED)
            self.root.update()

            # Well locations (IL, XL)
            wells = {
                'BULTE-1': {'il': 5110, 'xl': 5300},
                'HERWA-01': {'il': 5930, 'xl': 5320},
                'KASADE-01': {'il': 5200, 'xl': 4940},
            }

            # Open SEGY and extract data
            with segyio.open(str(segy_path), 'r', ignore_geometry=True) as segy:
                n_traces = segy.tracecount
                n_samples = len(segy.samples)
                sample_rate = segyio.tools.dt(segy) / 1000.0  # ms

                # Get survey geometry from headers
                il_min, il_max = 5047, 6047
                xl_min, xl_max = 4885, 7020

                # Extract a representative inline (through BULTE-1)
                target_il = 5110
                inline_traces = []
                inline_xls = []

                # Read traces for this inline
                for i in range(min(n_traces, 500000)):  # Limit for speed
                    try:
                        header = segy.header[i]
                        il = header.get(segyio.TraceField.INLINE_3D, 0)
                        xl = header.get(segyio.TraceField.CROSSLINE_3D, 0)

                        if il == target_il:
                            inline_traces.append(segy.trace[i])
                            inline_xls.append(xl)
                    except:
                        continue

                if not inline_traces:
                    # Fallback: just take sequential traces
                    step = max(1, n_traces // 500)
                    for i in range(0, min(n_traces, 500), step):
                        inline_traces.append(segy.trace[i])
                        inline_xls.append(xl_min + i * (xl_max - xl_min) // 500)

                # Convert to array
                inline_data = np.array(inline_traces).T  # (samples, traces)
                inline_xls = np.array(inline_xls)

                # Extract a time slice at ~1200ms (reservoir level)
                time_slice_ms = 1200
                sample_idx = int(time_slice_ms / sample_rate)

                # Sample time slice (subsample for speed)
                ts_step = max(1, n_traces // 10000)
                ts_data = []
                ts_ils = []
                ts_xls = []

                for i in range(0, min(n_traces, 100000), ts_step):
                    try:
                        header = segy.header[i]
                        il = header.get(segyio.TraceField.INLINE_3D, 0)
                        xl = header.get(segyio.TraceField.CROSSLINE_3D, 0)
                        if il > 0 and xl > 0 and sample_idx < n_samples:
                            ts_data.append(segy.trace[i][sample_idx])
                            ts_ils.append(il)
                            ts_xls.append(xl)
                    except:
                        continue

            # Create figure with 2 panels
            fig, axes = plt.subplots(1, 2, figsize=(14, 7), facecolor='#1a1a2e')

            # Panel 1: Inline section with well markers
            ax1 = axes[0]
            ax1.set_facecolor('#16213e')

            if len(inline_traces) > 0:
                time_axis = np.arange(n_samples) * sample_rate
                clip = np.percentile(np.abs(inline_data), 98)

                im1 = ax1.imshow(
                    inline_data,
                    aspect='auto',
                    extent=[inline_xls.min(), inline_xls.max(), time_axis[-1], time_axis[0]],
                    cmap='seismic',
                    vmin=-clip, vmax=clip
                )

                # Mark wells on inline
                for well_name, loc in wells.items():
                    if loc['il'] == target_il or abs(loc['il'] - target_il) < 50:
                        ax1.axvline(x=loc['xl'], color='lime', linewidth=2, linestyle='--')
                        ax1.text(loc['xl'], 100, well_name, color='lime', fontsize=9,
                                fontweight='bold', ha='center',
                                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

                # Mark time slice level
                ax1.axhline(y=time_slice_ms, color='yellow', linewidth=1.5, linestyle=':')
                ax1.text(inline_xls.min() + 50, time_slice_ms - 50, f'Time slice @ {time_slice_ms}ms',
                        color='yellow', fontsize=9)

            ax1.set_xlabel('Crossline', color='white', fontsize=11)
            ax1.set_ylabel('TWT (ms)', color='white', fontsize=11)
            ax1.set_title(f'Inline {target_il} - Seismic Section\n(through BULTE-1)', color='#e94560', fontsize=12, fontweight='bold')
            ax1.tick_params(colors='white')

            # Panel 2: Time slice with well markers
            ax2 = axes[1]
            ax2.set_facecolor('#16213e')

            if len(ts_data) > 0:
                # Create scatter plot for time slice
                scatter = ax2.scatter(ts_xls, ts_ils, c=ts_data, cmap='seismic',
                                     s=2, vmin=-np.percentile(np.abs(ts_data), 98),
                                     vmax=np.percentile(np.abs(ts_data), 98))

                # Mark wells
                for well_name, loc in wells.items():
                    ax2.scatter(loc['xl'], loc['il'], c='lime', s=200, marker='*',
                               edgecolors='white', linewidths=2, zorder=10)
                    ax2.annotate(well_name, (loc['xl'], loc['il']),
                                xytext=(10, 10), textcoords='offset points',
                                color='white', fontsize=10, fontweight='bold',
                                bbox=dict(boxstyle='round', facecolor='#e94560', alpha=0.9))

                # Mark the inline shown in panel 1
                ax2.axhline(y=target_il, color='cyan', linewidth=2, linestyle='--', label=f'IL {target_il}')

            ax2.set_xlabel('Crossline', color='white', fontsize=11)
            ax2.set_ylabel('Inline', color='white', fontsize=11)
            ax2.set_title(f'Time Slice @ {time_slice_ms}ms\n(Reservoir Level)', color='#e94560', fontsize=12, fontweight='bold')
            ax2.tick_params(colors='white')
            ax2.set_xlim(xl_min, xl_max)
            ax2.set_ylim(il_min, il_max)

            # Legend
            legend_elements = [
                Line2D([0], [0], marker='*', color='w', markerfacecolor='lime', markersize=15, label='Wells', linestyle='None'),
                Line2D([0], [0], color='cyan', linewidth=2, linestyle='--', label='Inline Section'),
                Line2D([0], [0], color='yellow', linewidth=2, linestyle=':', label='Time Slice Level'),
            ]
            ax2.legend(handles=legend_elements, loc='upper right', facecolor='#16213e', labelcolor='white')

            # Add info text
            fig.text(0.5, 0.02,
                    f'Bornu Chad Basin 3D Seismic | IL: {il_min}-{il_max} | XL: {xl_min}-{xl_max} | {n_traces:,} traces',
                    ha='center', color='white', fontsize=10)

            plt.tight_layout(rect=[0, 0.05, 1, 1])

            # Save
            output_dir = BASE_DIR / 'interpretation' / 'real_outputs'
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / 'seismic_3d_with_wells.png'
            fig.savefig(output_path, dpi=150, facecolor='#1a1a2e', edgecolor='none')
            plt.close(fig)

            # Refresh and display
            self.refresh_maps_list()
            self._display_map_preview(output_path)

            # Update summary
            self.interp_summary.config(state=tk.NORMAL)
            self.interp_summary.delete(1.0, tk.END)
            self.interp_summary.insert(tk.END, f"3D Seismic visualization generated!\n\n"
                                              f"Left: Inline {target_il} (vertical section)\n"
                                              f"Right: Time slice @ {time_slice_ms}ms (map view)\n\n"
                                              f"Wells shown: BULTE-1, HERWA-01, KASADE-01\n"
                                              f"Saved to: {output_path}")
            self.interp_summary.config(state=tk.DISABLED)

            messagebox.showinfo("Success", f"3D Seismic + Wells visualization generated!\n\nSaved to:\n{output_path}")

        except Exception as e:
            import traceback
            error_msg = f"Could not generate seismic visualization:\n{str(e)}\n\n{traceback.format_exc()}"
            messagebox.showerror("Error", error_msg)
            self.interp_summary.config(state=tk.NORMAL)
            self.interp_summary.delete(1.0, tk.END)
            self.interp_summary.insert(tk.END, error_msg)
            self.interp_summary.config(state=tk.DISABLED)

    def generate_survey_basemap(self):
        """Generate a basemap showing 3D survey extent and well locations"""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle, Circle
        from matplotlib.lines import Line2D

        try:
            # Survey parameters (from 3D seismic)
            il_min, il_max = 5047, 6047
            xl_min, xl_max = 4885, 7020

            # Well locations (IL, XL coordinates)
            wells = {
                'BULTE-1': {'il': 5110, 'xl': 5300, 'in_3d': True},
                'HERWA-01': {'il': 5930, 'xl': 5320, 'in_3d': True},
                'KASADE-01': {'il': 5200, 'xl': 4940, 'in_3d': True},
                'MASU-1': {'il': 4800, 'xl': 5100, 'in_3d': False},  # Outside
                'NGAMMAEAST-1': {'il': 4500, 'xl': 4500, 'in_3d': False},  # Outside
                'NGORNORTH-1': {'il': 4600, 'xl': 4700, 'in_3d': False},  # Outside
            }

            # Load fault data if available
            faults = []
            fault_file = BASE_DIR / 'interpretation' / 'real_outputs' / 'interpretation_results.json'
            if fault_file.exists():
                with open(fault_file) as f:
                    data = json.load(f)
                    faults = data.get('faults', [])[:20]  # Top 20 faults

            # Create figure
            fig, ax = plt.subplots(figsize=(12, 10), facecolor='#1a1a2e')
            ax.set_facecolor('#16213e')

            # Draw 3D survey extent
            survey_rect = Rectangle(
                (xl_min, il_min),
                xl_max - xl_min,
                il_max - il_min,
                linewidth=3,
                edgecolor='#4ecca3',
                facecolor='#4ecca320',
                label='3D Survey (~1,955 km²)'
            )
            ax.add_patch(survey_rect)

            # Draw grid lines
            for il in range(il_min, il_max + 1, 200):
                ax.axhline(y=il, color='#0f3460', linewidth=0.5, alpha=0.5)
            for xl in range(xl_min, xl_max + 1, 400):
                ax.axvline(x=xl, color='#0f3460', linewidth=0.5, alpha=0.5)

            # Draw faults
            for fault in faults:
                if 'representative_xl' in fault and 'representative_il' in fault:
                    xl = fault.get('representative_xl', (xl_min + xl_max) / 2)
                    il = fault.get('representative_il', (il_min + il_max) / 2)
                    # Draw as short line segment
                    strike = fault.get('avg_strike_deg', 45)
                    import math
                    dx = 100 * math.cos(math.radians(strike))
                    dy = 100 * math.sin(math.radians(strike))
                    ax.plot([xl - dx, xl + dx], [il - dy, il + dy],
                           color='yellow', linewidth=1.5, alpha=0.7)

            # Plot wells
            for well_name, loc in wells.items():
                il, xl = loc['il'], loc['xl']
                in_3d = loc['in_3d']

                if in_3d:
                    color = '#e94560'
                    marker = 'o'
                    size = 150
                else:
                    color = '#ffc107'
                    marker = 's'
                    size = 100

                ax.scatter(xl, il, c=color, s=size, marker=marker, edgecolors='white', linewidths=2, zorder=5)
                ax.annotate(
                    well_name,
                    (xl, il),
                    xytext=(10, 10),
                    textcoords='offset points',
                    fontsize=10,
                    fontweight='bold',
                    color='white',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8)
                )

            # Set labels and title
            ax.set_xlabel('Crossline', fontsize=12, color='white')
            ax.set_ylabel('Inline', fontsize=12, color='white')
            ax.set_title(
                'Bornu Chad Basin - 3D Survey Basemap\nAI-Augmented Seismic Interpretation',
                fontsize=14,
                fontweight='bold',
                color='#e94560'
            )

            # Set axis limits with padding
            ax.set_xlim(xl_min - 500, xl_max + 500)
            ax.set_ylim(il_min - 300, il_max + 300)

            # Style ticks
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_color('#4ecca3')

            # Legend
            legend_elements = [
                Rectangle((0, 0), 1, 1, facecolor='#4ecca320', edgecolor='#4ecca3', linewidth=2, label='3D Survey Extent'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='#e94560', markersize=10, label='Wells (in 3D)', linestyle='None'),
                Line2D([0], [0], marker='s', color='w', markerfacecolor='#ffc107', markersize=10, label='Wells (outside 3D)', linestyle='None'),
            ]
            if faults:
                legend_elements.append(Line2D([0], [0], color='yellow', linewidth=2, label=f'Faults ({len(faults)} shown)'))

            ax.legend(
                handles=legend_elements,
                loc='upper right',
                facecolor='#16213e',
                edgecolor='#4ecca3',
                labelcolor='white'
            )

            # Add survey info text
            info_text = (
                f"Survey: IL {il_min}-{il_max}, XL {xl_min}-{xl_max}\n"
                f"Traces: ~1.84 million | Area: ~1,955 km²\n"
                f"Wells in 3D: 3 | Wells outside: 3"
            )
            ax.text(
                0.02, 0.02,
                info_text,
                transform=ax.transAxes,
                fontsize=9,
                color='white',
                verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='#0f3460', alpha=0.9)
            )

            plt.tight_layout()

            # Save basemap - use absolute path
            output_dir = BASE_DIR / 'interpretation' / 'real_outputs'
            output_dir.mkdir(parents=True, exist_ok=True)
            basemap_path = output_dir / 'survey_basemap.png'
            fig.savefig(basemap_path, dpi=150, facecolor='#1a1a2e', edgecolor='none')
            plt.close(fig)

            # Refresh maps list and show the basemap
            self.refresh_maps_list()

            # Display the basemap in preview
            self._display_map_preview(basemap_path)

            messagebox.showinfo("Success", f"Survey basemap generated!\n\nSaved to: {basemap_path}")

        except Exception as e:
            messagebox.showerror("Error", f"Could not generate basemap:\n{str(e)}")

    def run_synthetic_workflow(self):
        """Run synthetic seismogram generation"""
        self.notebook.select(self.process_tab)
        self.run_processing_step(5)

    def run_horizon_workflow(self):
        """Run horizon mapping workflow"""
        self.notebook.select(self.process_tab)
        self.run_processing_step(6)

    def run_attribute_workflow(self):
        """Run seismic attributes workflow"""
        self.notebook.select(self.process_tab)
        self.run_processing_step(7)

    def run_full_interpretation(self):
        """Run complete interpretation workflow"""
        self.notebook.select(self.process_tab)
        self.run_processing_step(8)

    def open_interpretation_folder(self):
        """Open the interpretation output folder"""
        import subprocess

        # Priority order: real_outputs (PhD critical), then others
        possible_paths = [
            BASE_DIR / 'interpretation' / 'real_outputs',  # Step 10 outputs (PhD critical)
            BASE_DIR / 'interpretation' / 'synthetic_outputs',
            BASE_DIR / 'interpretation' / 'structure_outputs',
            BASE_DIR / 'interpretation' / 'attribute_outputs',
            BASE_DIR / 'outputs' / 'interpretation',
            BASE_DIR / 'outputs' / 'figures',
        ]

        for path in possible_paths:
            if path.exists() and any(path.iterdir()):
                subprocess.Popen(['explorer', str(path.resolve())])
                return

        messagebox.showinfo("Info", f"No interpretation outputs found yet.\n\nExpected location:\n{possible_paths[0]}\n\nRun Step 10 (REAL Interpretation) first.")

    def refresh_maps_list(self):
        """Refresh the list of generated maps"""
        self.maps_listbox.delete(0, tk.END)

        # Search for map images - real_outputs is priority (PhD critical)
        # Use BASE_DIR for absolute paths
        map_dirs = [
            BASE_DIR / 'interpretation' / 'real_outputs',  # Step 10 outputs (PhD critical)
            BASE_DIR / 'interpretation' / 'synthetic_outputs',
            BASE_DIR / 'interpretation' / 'structure_outputs',
            BASE_DIR / 'interpretation' / 'attribute_outputs',
            BASE_DIR / 'outputs' / 'interpretation',
            BASE_DIR / 'outputs' / 'figures',
        ]

        found_maps = []
        for map_dir in map_dirs:
            if map_dir.exists():
                for ext in ['*.png', '*.jpg', '*.jpeg']:
                    found_maps.extend(map_dir.glob(ext))
                print(f"[DEBUG] Found {len(list(map_dir.glob('*.png')))} PNG files in {map_dir}")

        # Sort by modification time (newest first)
        found_maps.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        print(f"[DEBUG] Total maps found: {len(found_maps)}")

        for map_path in found_maps[:50]:  # Limit to 50 most recent
            display_name = f"{map_path.parent.name}/{map_path.name}"
            self.maps_listbox.insert(tk.END, display_name)
            print(f"[DEBUG] Added to list: {display_name}")

        # Store paths for selection
        self.map_paths = found_maps[:50]

        if not found_maps:
            self.maps_listbox.insert(tk.END, "(No maps found)")
            self.maps_listbox.insert(tk.END, f"Searched in: {map_dirs[0]}")

        # Update summary
        self._update_interpretation_summary()

    def on_map_select(self, event):
        """Handle map selection from list"""
        selection = self.maps_listbox.curselection()
        if selection and hasattr(self, 'map_paths'):
            idx = selection[0]
            if idx < len(self.map_paths):
                map_path = self.map_paths[idx]
                self._display_map_preview(map_path)

    def _display_map_preview(self, map_path):
        """Display selected map in preview area"""
        try:
            from PIL import Image, ImageTk

            img = Image.open(map_path)
            # Resize to fit preview area
            max_size = (600, 400)
            img.thumbnail(max_size, Image.Resampling.LANCZOS)

            photo = ImageTk.PhotoImage(img)
            self.map_preview_label.configure(image=photo, text='')
            self.map_preview_label.image = photo  # Keep reference
        except Exception as e:
            self.map_preview_label.configure(
                image='',
                text=f"Could not load image:\n{map_path.name}\n\nError: {str(e)}"
            )

    def _update_interpretation_summary(self):
        """Update interpretation summary based on available outputs"""
        summary_lines = []

        # Check for REAL interpretation results (Step 10 - PhD critical)
        # Use BASE_DIR for absolute path
        real_results_file = BASE_DIR / 'interpretation' / 'real_outputs' / 'interpretation_results.json'
        if real_results_file.exists():
            try:
                with open(real_results_file) as f:
                    real_data = json.load(f)

                # Well ties
                if 'well_ties' in real_data:
                    ties = real_data['well_ties']
                    n_wells = len(ties)
                    avg_corr = sum(t.get('correlation', 0) for t in ties.values()) / max(n_wells, 1)
                    summary_lines.append(f"Well Ties: {n_wells} wells, avg r={avg_corr:.3f}")

                # Horizons
                if 'horizons' in real_data:
                    horizons = real_data['horizons']
                    n_horizons = len(horizons)
                    summary_lines.append(f"Horizons: {n_horizons} mapped (100% coverage)")

                # Attributes
                if 'attributes' in real_data:
                    attrs = real_data['attributes']
                    n_dhi = sum(a.get('dhi_anomaly_count', 0) for a in attrs.values())
                    summary_lines.append(f"DHI Anomalies: {n_dhi:,} points detected")

                # Faults
                if 'faults' in real_data:
                    faults = real_data['faults']
                    n_faults = len(faults)
                    total_segments = real_data.get('fault_summary', {}).get('total_segments_detected', n_faults)
                    summary_lines.append(f"Faults: {n_faults} significant (of {total_segments} detected)")

                # Volumetrics
                if 'volumetrics' in real_data:
                    vol = real_data['volumetrics']
                    p50 = vol.get('p50_mmstb', 0)
                    if p50 > 0:
                        summary_lines.append(f"STOIIP P50: {p50:.1f} MMSTB")

            except Exception as e:
                summary_lines.append(f"Error reading results: {str(e)}")

        # Check for synthetic results (legacy)
        synth_file = BASE_DIR / 'interpretation' / 'synthetic_outputs' / 'synthetic_results.json'
        if synth_file.exists() and not real_results_file.exists():
            try:
                with open(synth_file) as f:
                    synth_data = json.load(f)
                n_wells = len(synth_data.get('synthetics', {}))
                n_tops = len(synth_data.get('formation_tops', {}))
                summary_lines.append(f"Synthetics: {n_wells} wells tied, {n_tops} formation tops")
            except:
                pass

        # Check real_outputs directory for images
        real_output_dir = BASE_DIR / 'interpretation' / 'real_outputs'
        if real_output_dir.exists():
            n_maps = len(list(real_output_dir.glob('*.png')))
            if n_maps > 0:
                summary_lines.append(f"Maps Generated: {n_maps} images")

        # Check for maps
        if hasattr(self, 'map_paths') and self.map_paths:
            summary_lines.append(f"Total Available: {len(self.map_paths)} images")

        if not summary_lines:
            summary_lines.append("No interpretation outputs found yet.")
            summary_lines.append("")
            summary_lines.append("Run Step 10 (REAL Interpretation) from Processing tab")
            summary_lines.append("or use Quick Actions above.")

        self.interp_summary.config(state=tk.NORMAL)
        self.interp_summary.delete(1.0, tk.END)
        self.interp_summary.insert(tk.END, '\n'.join(summary_lines))
        self.interp_summary.config(state=tk.DISABLED)

    def _create_welllogs_tab(self):
        """Create Well Logs & Formation Picks tab"""
        # Main container
        main_frame = tk.Frame(self.welllogs_tab, bg=self.colors['bg'], padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        tk.Label(
            main_frame,
            text="Well Logs & Formation Picks",
            font=('Segoe UI', 18, 'bold'),
            bg=self.colors['bg'],
            fg=self.colors['accent']
        ).pack(anchor=tk.W)

        tk.Label(
            main_frame,
            text="View well logs, pick formation tops, and manage picks for synthetic seismogram generation",
            font=('Segoe UI', 10),
            bg=self.colors['bg'],
            fg=self.colors['fg']
        ).pack(anchor=tk.W, pady=(5, 20))

        # Split view
        content = tk.Frame(main_frame, bg=self.colors['bg'])
        content.pack(fill=tk.BOTH, expand=True)

        # Left: Well selection and formation picks
        left_frame = tk.Frame(content, bg=self.colors['surface'], width=350)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_frame.pack_propagate(False)

        # Well selection
        tk.Label(
            left_frame,
            text="Select Well",
            font=('Segoe UI', 12, 'bold'),
            bg=self.colors['surface'],
            fg=self.colors['fg']
        ).pack(anchor=tk.W, padx=15, pady=(15, 5))

        self.well_listbox = tk.Listbox(
            left_frame,
            font=('Segoe UI', 10),
            bg=self.colors['overlay'],
            fg=self.colors['fg'],
            selectbackground=self.colors['accent'],
            relief=tk.FLAT,
            height=6
        )
        self.well_listbox.pack(fill=tk.X, padx=15, pady=5)
        self.well_listbox.bind('<<ListboxSelect>>', self.on_well_select_for_logs)

        # Populate wells
        self._populate_well_list()

        tk.Button(
            left_frame,
            text="Display Well Logs",
            font=('Segoe UI', 10),
            bg=self.colors['accent'],
            fg='white',
            relief=tk.FLAT,
            command=self.display_selected_well_logs
        ).pack(fill=tk.X, padx=15, pady=5)

        # Separator
        tk.Frame(left_frame, bg=self.colors['overlay'], height=1).pack(fill=tk.X, padx=15, pady=10)

        # Formation Picks Section
        tk.Label(
            left_frame,
            text="Formation Picks (depth in meters)",
            font=('Segoe UI', 12, 'bold'),
            bg=self.colors['surface'],
            fg=self.colors['fg']
        ).pack(anchor=tk.W, padx=15, pady=(5, 10))

        # Formation input fields
        self.formation_entries = {}
        formations = [
            ('Chad_Fm', 'Chad Formation'),
            ('Fika_Shale', 'Fika Shale'),
            ('Gongila_Fm', 'Gongila Formation'),
            ('Bima_Sst', 'Bima Sandstone')
        ]

        for fm_key, fm_name in formations:
            frame = tk.Frame(left_frame, bg=self.colors['surface'])
            frame.pack(fill=tk.X, padx=15, pady=2)

            tk.Label(
                frame,
                text=fm_name + ":",
                font=('Segoe UI', 9),
                bg=self.colors['surface'],
                fg=self.colors['fg'],
                width=15,
                anchor=tk.W
            ).pack(side=tk.LEFT)

            entry = tk.Entry(
                frame,
                font=('Segoe UI', 10),
                bg=self.colors['overlay'],
                fg=self.colors['fg'],
                relief=tk.FLAT,
                width=10
            )
            entry.pack(side=tk.LEFT, padx=5)
            self.formation_entries[fm_key] = entry

            tk.Label(
                frame,
                text="m",
                font=('Segoe UI', 9),
                bg=self.colors['surface'],
                fg=self.colors['fg']
            ).pack(side=tk.LEFT)

        # Buttons for picks
        btn_frame = tk.Frame(left_frame, bg=self.colors['surface'])
        btn_frame.pack(fill=tk.X, padx=15, pady=10)

        tk.Button(
            btn_frame,
            text="Save Picks",
            font=('Segoe UI', 10),
            bg=self.colors['success'],
            fg='white',
            relief=tk.FLAT,
            command=self.save_formation_picks
        ).pack(side=tk.LEFT, padx=(0, 5))

        tk.Button(
            btn_frame,
            text="Load from Literature",
            font=('Segoe UI', 10),
            bg=self.colors['overlay'],
            fg=self.colors['fg'],
            relief=tk.FLAT,
            command=self.load_literature_picks
        ).pack(side=tk.LEFT)

        # Separator
        tk.Frame(left_frame, bg=self.colors['overlay'], height=1).pack(fill=tk.X, padx=15, pady=10)

        # Picking Guide
        tk.Label(
            left_frame,
            text="Quick Picking Guide",
            font=('Segoe UI', 11, 'bold'),
            bg=self.colors['surface'],
            fg=self.colors['fg']
        ).pack(anchor=tk.W, padx=15, pady=(5, 5))

        guide_text = """Chad Fm: Low GR at surface
Fika Shale: High GR (>100 API)
Gongila Fm: Variable GR below Fika
Bima Sst: Low GR (<60 API), clean sand

Look for GR changes and
resistivity trends!"""

        tk.Label(
            left_frame,
            text=guide_text,
            font=('Consolas', 9),
            bg=self.colors['surface'],
            fg=self.colors['fg'],
            justify=tk.LEFT
        ).pack(anchor=tk.W, padx=15, pady=5)

        # Right: Log display area
        right_frame = tk.Frame(content, bg=self.colors['surface'])
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        tk.Label(
            right_frame,
            text="Well Log Display",
            font=('Segoe UI', 12, 'bold'),
            bg=self.colors['surface'],
            fg=self.colors['fg']
        ).pack(anchor=tk.W, padx=15, pady=(15, 10))

        # Canvas for matplotlib figure
        self.log_display_frame = tk.Frame(right_frame, bg=self.colors['overlay'])
        self.log_display_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))

        self.log_placeholder = tk.Label(
            self.log_display_frame,
            text="Select a well and click 'Display Well Logs'\nto view GR, Resistivity, Density, and Sonic logs\n\nFormation tops will be marked on the display",
            font=('Segoe UI', 12),
            bg=self.colors['overlay'],
            fg=self.colors['fg'],
            justify=tk.CENTER
        )
        self.log_placeholder.pack(fill=tk.BOTH, expand=True, pady=50)

        self.current_log_canvas = None

    def _populate_well_list(self):
        """Populate the well list from available LAS files"""
        self.well_listbox.delete(0, tk.END)

        # Get LAS directory from config or use current directory
        if CONFIG_AVAILABLE:
            config = get_config()
            las_dir = Path(config.well_logs_directory) if config.well_logs_directory else Path()
        else:
            las_dir = Path()
        wells = []

        if las_dir.exists():
            for f in las_dir.glob("*.las"):
                wells.append(f.stem)
            for f in las_dir.glob("*.LAS"):
                if f.stem not in wells:
                    wells.append(f.stem)

        for well in sorted(wells):
            self.well_listbox.insert(tk.END, well)

    def on_well_select_for_logs(self, event):
        """Handle well selection - load existing picks"""
        selection = self.well_listbox.curselection()
        if not selection:
            return

        well_name = self.well_listbox.get(selection[0])
        self._load_picks_for_well(well_name)

    def _load_picks_for_well(self, well_name: str):
        """Load existing formation picks for a well"""
        tops_file = BASE_DIR / "formation_tops.json"

        # Clear entries
        for entry in self.formation_entries.values():
            entry.delete(0, tk.END)

        if tops_file.exists():
            try:
                with open(tops_file, 'r') as f:
                    data = json.load(f)

                wells_data = data.get('wells', {})
                well_data = wells_data.get(well_name, {})

                # Try fuzzy match
                if not well_data:
                    for wn, wd in wells_data.items():
                        if wn.replace("-", "").upper() == well_name.replace("-", "").upper():
                            well_data = wd
                            break

                for fm_key, entry in self.formation_entries.items():
                    value = well_data.get(fm_key)
                    if value is not None:
                        entry.insert(0, str(value))

            except Exception as e:
                print(f"Error loading picks: {e}")

    def display_selected_well_logs(self):
        """Display well logs for the selected well"""
        selection = self.well_listbox.curselection()
        if not selection:
            messagebox.showinfo("Info", "Please select a well first")
            return

        well_name = self.well_listbox.get(selection[0])

        try:
            # Import the viewer
            from formation_tops_manager import WellLogViewer

            viewer = WellLogViewer()
            fig = viewer.plot_well_logs(well_name, figsize=(12, 8))

            # Clear previous display
            if self.current_log_canvas:
                self.current_log_canvas.get_tk_widget().destroy()
            self.log_placeholder.pack_forget()

            # Embed matplotlib figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            self.current_log_canvas = FigureCanvasTkAgg(fig, self.log_display_frame)
            self.current_log_canvas.draw()
            self.current_log_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        except Exception as e:
            messagebox.showerror("Error", f"Could not display logs for {well_name}:\n{str(e)}")
            import traceback
            traceback.print_exc()

    def save_formation_picks(self):
        """Save formation picks to JSON file"""
        selection = self.well_listbox.curselection()
        if not selection:
            messagebox.showinfo("Info", "Please select a well first")
            return

        well_name = self.well_listbox.get(selection[0])
        tops_file = BASE_DIR / "formation_tops.json"

        # Load existing data
        if tops_file.exists():
            with open(tops_file, 'r') as f:
                data = json.load(f)
        else:
            data = {"metadata": {}, "wells": {}}

        # Update well data
        if well_name not in data['wells']:
            data['wells'][well_name] = {}

        for fm_key, entry in self.formation_entries.items():
            value = entry.get().strip()
            if value:
                try:
                    data['wells'][well_name][fm_key] = float(value)
                except ValueError:
                    pass

        data['wells'][well_name]['last_updated'] = datetime.now().isoformat()

        # Save
        with open(tops_file, 'w') as f:
            json.dump(data, f, indent=2)

        messagebox.showinfo("Success", f"Formation picks saved for {well_name}")

    def load_literature_picks(self):
        """Load formation picks from literature for selected well"""
        selection = self.well_listbox.curselection()
        if not selection:
            messagebox.showinfo("Info", "Please select a well first")
            return

        well_name = self.well_listbox.get(selection[0])

        # Literature values
        literature = {
            'BULTE-1': {'Chad_Fm': 380, 'Fika_Shale': 720, 'Gongila_Fm': 1050, 'Bima_Sst': 1280},
            'HERWA-01': {'Chad_Fm': 450, 'Fika_Shale': 890, 'Gongila_Fm': 1350, 'Bima_Sst': 1680},
            'MASU-1': {'Fika_Shale': 2025, 'Gongila_Fm': 2450, 'Bima_Sst': 2850},
            'KASADE-01': {'Chad_Fm': 350, 'Fika_Shale': 680, 'Gongila_Fm': 980, 'Bima_Sst': 1250},
            'NGAMMAEAST-1': {'Chad_Fm': 320, 'Fika_Shale': 850, 'Gongila_Fm': 1600, 'Bima_Sst': 2100},
            'NGORNORTH-1': {'Chad_Fm': 280, 'Fika_Shale': 520, 'Gongila_Fm': 850, 'Bima_Sst': 1100}
        }

        # Find matching well
        well_data = None
        for wn, wd in literature.items():
            if wn.replace("-", "").upper() == well_name.replace("-", "").upper():
                well_data = wd
                break

        if well_data:
            # Clear and populate entries
            for fm_key, entry in self.formation_entries.items():
                entry.delete(0, tk.END)
                value = well_data.get(fm_key)
                if value is not None:
                    entry.insert(0, str(value))
            messagebox.showinfo("Loaded", f"Literature picks loaded for {well_name}\n\nSource: Avbovbo (1986), Okosun (1995), NNPC Reports")
        else:
            messagebox.showwarning("Not Found", f"No literature values found for {well_name}")

    def _create_image_tab(self):
        """Create image interpretation tab"""
        # Main container
        main_frame = tk.Frame(self.image_tab, bg=self.colors['bg'], padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        tk.Label(
            main_frame,
            text="Seismic Image Interpretation",
            font=('Segoe UI', 18, 'bold'),
            bg=self.colors['bg'],
            fg=self.colors['accent']
        ).pack(anchor=tk.W)

        tk.Label(
            main_frame,
            text="Upload seismic sections, maps, or attribute displays for AI interpretation using Llava vision model",
            font=('Segoe UI', 10),
            bg=self.colors['bg'],
            fg=self.colors['fg']
        ).pack(anchor=tk.W, pady=(5, 20))

        # Split view
        content = tk.Frame(main_frame, bg=self.colors['bg'])
        content.pack(fill=tk.BOTH, expand=True)

        # Left: Image display
        left_frame = tk.Frame(content, bg=self.colors['surface'], width=500)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        self.image_label = tk.Label(
            left_frame,
            text="No image loaded\n\nClick 'Load Image' to select a seismic image\n(PNG, JPG, or screenshot)",
            font=('Segoe UI', 12),
            bg=self.colors['overlay'],
            fg=self.colors['fg'],
            width=50,
            height=20
        )
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Image buttons
        img_btn_frame = tk.Frame(left_frame, bg=self.colors['surface'])
        img_btn_frame.pack(fill=tk.X, padx=20, pady=(0, 20))

        tk.Button(
            img_btn_frame,
            text="Load Image",
            font=('Segoe UI', 11),
            bg=self.colors['accent'],
            fg='white',
            relief=tk.FLAT,
            padx=20,
            command=self.load_image
        ).pack(side=tk.LEFT)

        tk.Button(
            img_btn_frame,
            text="Load from Outputs",
            font=('Segoe UI', 11),
            bg=self.colors['overlay'],
            fg=self.colors['fg'],
            relief=tk.FLAT,
            padx=20,
            command=self.load_output_image
        ).pack(side=tk.LEFT, padx=10)

        self.current_image_path = None

        # Right: Interpretation
        right_frame = tk.Frame(content, bg=self.colors['bg'], width=500)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Prompt input
        tk.Label(
            right_frame,
            text="Additional Instructions (optional):",
            font=('Segoe UI', 11, 'bold'),
            bg=self.colors['bg'],
            fg=self.colors['fg']
        ).pack(anchor=tk.W)

        self.image_prompt = tk.Entry(
            right_frame,
            font=('Consolas', 11),
            bg=self.colors['surface'],
            fg=self.colors['fg'],
            insertbackground=self.colors['fg'],
            relief=tk.FLAT
        )
        self.image_prompt.pack(fill=tk.X, pady=(5, 10), ipady=8)
        self.image_prompt.insert(0, "Focus on hydrocarbon indicators and structural features")

        tk.Button(
            right_frame,
            text="Interpret Image",
            font=('Segoe UI', 12, 'bold'),
            bg=self.colors['success'],
            fg='white',
            relief=tk.FLAT,
            padx=30,
            pady=8,
            command=self.interpret_current_image
        ).pack(anchor=tk.W)

        # Interpretation output
        tk.Label(
            right_frame,
            text="AI Interpretation:",
            font=('Segoe UI', 11, 'bold'),
            bg=self.colors['bg'],
            fg=self.colors['fg']
        ).pack(anchor=tk.W, pady=(20, 5))

        self.image_interpretation = scrolledtext.ScrolledText(
            right_frame,
            wrap=tk.WORD,
            font=('Consolas', 10),
            bg=self.colors['surface'],
            fg=self.colors['fg'],
            relief=tk.FLAT,
            padx=10,
            pady=10
        )
        self.image_interpretation.pack(fill=tk.BOTH, expand=True)

    def _create_seismic_viewer_tab(self):
        """Create interactive seismic viewer tab with AI interpretation"""
        if not SEISMIC_VIEWER_AVAILABLE:
            # Show fallback message
            tk.Label(
                self.seismic_viewer_tab,
                text="Seismic Viewer not available.\n\nThe seismic_viewer.py module could not be loaded.\nCheck that all dependencies (segyio, matplotlib) are installed.",
                font=('Segoe UI', 14),
                bg=self.colors['bg'],
                fg=self.colors['warning'],
                justify=tk.CENTER
            ).pack(expand=True)
            return

        # Create seismic viewer
        self.seismic_viewer = SeismicViewerGUI(
            self.seismic_viewer_tab,
            ollama_client=self.ollama,
            colors=self.colors
        )

        # Store reference to loader for well tie panel
        self.seismic_loader = self.seismic_viewer.loader

    def _create_well_tie_tab(self):
        """Create well-to-seismic tie validation tab"""
        if not SEISMIC_VIEWER_AVAILABLE:
            # Show fallback message
            tk.Label(
                self.well_tie_tab,
                text="Well Tie Validation not available.\n\nThe seismic_viewer.py module could not be loaded.",
                font=('Segoe UI', 14),
                bg=self.colors['bg'],
                fg=self.colors['warning'],
                justify=tk.CENTER
            ).pack(expand=True)
            return

        # Create title
        title_frame = tk.Frame(self.well_tie_tab, bg=self.colors['bg'])
        title_frame.pack(fill=tk.X, padx=20, pady=10)

        tk.Label(
            title_frame,
            text="Well-to-Seismic Tie Validation",
            font=('Segoe UI', 18, 'bold'),
            bg=self.colors['bg'],
            fg=self.colors['accent']
        ).pack(side=tk.LEFT)

        tk.Label(
            title_frame,
            text="PhD Critical Step: Validate your horizon picks with well control",
            font=('Segoe UI', 10),
            bg=self.colors['bg'],
            fg=self.colors['fg']
        ).pack(side=tk.LEFT, padx=20)

        # Use the seismic loader from the viewer if available
        if hasattr(self, 'seismic_loader'):
            loader = self.seismic_loader
        else:
            loader = SeismicDataLoader()

        # Create well tie validator
        self.well_tie_validator = WellTieValidator(loader)

        # Create well tie panel
        self.well_tie_panel = WellTiePanel(
            self.well_tie_tab,
            self.well_tie_validator,
            colors=self.colors
        )

    def _create_status_bar(self):
        """Create status bar"""
        self.status_bar = tk.Frame(self.root, bg=self.colors['surface'], height=30)
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        self.status_bar.pack_propagate(False)

        self.status_label = tk.Label(
            self.status_bar,
            text="Ready",
            font=('Segoe UI', 9),
            bg=self.colors['surface'],
            fg=self.colors['fg']
        )
        self.status_label.pack(side=tk.LEFT, padx=10)

        completed = len(self.state.state['completed_steps'])
        self.progress_status = tk.Label(
            self.status_bar,
            text=f"Steps: {completed}/8 completed",
            font=('Segoe UI', 9),
            bg=self.colors['surface'],
            fg=self.colors['accent']
        )
        self.progress_status.pack(side=tk.RIGHT, padx=10)

    # ==========================================================================
    # HELPER METHODS
    # ==========================================================================

    def _show_welcome(self):
        self.add_chat_message("system", "Welcome to Seismic AI Assistant v2.0!")
        self.add_chat_message("system", f"Ollama: {'Connected' if self.ollama.is_connected else 'Not Connected'}")
        if self.ollama.is_connected:
            self.add_chat_message("system", f"Model: {self.ollama.model}")
            if self.ollama.has_vision:
                self.add_chat_message("system", "Vision model (llava) available for image interpretation!")

        # Check GPU status
        try:
            import cupy as cp
            device = cp.cuda.Device()
            mem = device.mem_info
            gpu_msg = f"GPU: Enabled - {mem[1]/1024**3:.1f} GB VRAM ({mem[0]/1024**3:.1f} GB free)"
            self.add_chat_message("system", gpu_msg)
        except:
            self.add_chat_message("system", "GPU: Not available (install cupy-cuda12x for acceleration)")

        self.add_chat_message("system", "\nI can help you with:\n- Processing seismic data (GPU accelerated!)\n- Recommending drilling locations\n- Calculating reserves (STOIIP)\n- Interpreting seismic images\n- Generating reports")

    def check_ollama_connection(self):
        if self.ollama.check_connection():
            self.connection_label.config(text="Connected", fg=self.colors['success'])
            self.model_combo['values'] = [m for m in self.ollama.available_models if "embed" not in m.lower()]
            self.model_var.set(self.ollama.model)
        else:
            self.connection_label.config(text="Not Connected", fg=self.colors['error'])

    def on_model_change(self, event):
        self.ollama.model = self.model_var.get()
        self.add_chat_message("system", f"Model changed to: {self.ollama.model}")

    def add_chat_message(self, sender: str, message: str):
        self.chat_display.config(state=tk.NORMAL)
        timestamp = datetime.now().strftime("%H:%M")

        tags = {'user': 'user', 'ai': 'ai', 'system': 'system', 'error': 'error'}
        tag = tags.get(sender, 'system')
        prefix = f"\n[{timestamp}] {sender.upper()}: "

        self.chat_display.insert(tk.END, prefix, tag)
        self.chat_display.insert(tk.END, message + "\n")
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)

    def terminal_print(self, message: str, tag: str = None):
        self.terminal.config(state=tk.NORMAL)
        if tag:
            self.terminal.insert(tk.END, message + "\n", tag)
        else:
            self.terminal.insert(tk.END, message + "\n")
        self.terminal.see(tk.END)
        self.terminal.config(state=tk.DISABLED)

    def clear_terminal(self):
        self.terminal.config(state=tk.NORMAL)
        self.terminal.delete(1.0, tk.END)
        self.terminal.config(state=tk.DISABLED)

    def browse_file(self, entry: tk.Entry, filetypes: list):
        path = filedialog.askopenfilename(filetypes=filetypes)
        if path:
            entry.delete(0, tk.END)
            entry.insert(0, path)

    def browse_folder(self, entry: tk.Entry):
        path = filedialog.askdirectory()
        if path:
            entry.delete(0, tk.END)
            entry.insert(0, path)

    def save_data_config(self):
        for key, entry in self.data_entries.items():
            self.state.set_data_path(key, entry.get())

        # Also update pipeline_config.json
        config_file = BASE_DIR / "pipeline_config.json"
        config = {}
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)

        config["input_segy_3d"] = self.data_entries["segy_3d"].get()
        config["input_2d_directory"] = self.data_entries["segy_2d_folder"].get()
        config["well_data"] = self.data_entries["well_folder"].get()
        config["well_header"] = self.data_entries["well_header"].get()
        config["output_dir"] = self.data_entries["output_folder"].get()

        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

        messagebox.showinfo("Saved", "Data configuration saved successfully!")

    def validate_data(self):
        issues = []

        segy_3d = self.data_entries["segy_3d"].get()
        if segy_3d and not Path(segy_3d).exists():
            issues.append(f"3D SEGY file not found: {segy_3d}")

        segy_2d = self.data_entries["segy_2d_folder"].get()
        if segy_2d and not Path(segy_2d).exists():
            issues.append(f"2D folder not found: {segy_2d}")

        wells = self.data_entries["well_folder"].get()
        if wells and not Path(wells).exists():
            issues.append(f"Well folder not found: {wells}")

        if issues:
            messagebox.showwarning("Validation Issues", "\n".join(issues))
        else:
            messagebox.showinfo("Validation", "All data paths are valid!")

    def on_enter(self, event):
        if not (event.state & 0x1):  # Shift not pressed
            self.send_message()
            return 'break'

    def send_message(self):
        message = self.chat_input.get("1.0", tk.END).strip()
        if not message:
            return

        self.chat_input.delete("1.0", tk.END)
        self.add_chat_message("user", message)
        self.status_label.config(text="AI thinking...")
        self.send_btn.config(state=tk.DISABLED)

        def get_response():
            response = self.interpreter.interpret_query(message)
            self.root.after(0, lambda: self._display_response(response))

        threading.Thread(target=get_response, daemon=True).start()

    def _display_response(self, response: str):
        self.add_chat_message("ai", response)
        self.status_label.config(text="Ready")
        self.send_btn.config(state=tk.NORMAL)

    def quick_query(self, query: str):
        self.chat_input.delete("1.0", tk.END)
        self.chat_input.insert("1.0", query)
        self.send_message()

    def run_processing_step(self, step_num: int):
        step_info = PROCESSING_STEPS.get(step_num)
        if not step_info:
            return

        script_path = BASE_DIR / step_info["folder"] / step_info["script"]
        if not script_path.exists():
            messagebox.showerror("Error", f"Script not found: {script_path}")
            return

        # Switch to processing tab
        self.notebook.select(2)

        # Update UI
        self.terminal_print(f"\n{'='*60}", 'info')
        self.terminal_print(f"Starting Step {step_num}: {step_info['name']}", 'info')
        self.terminal_print(f"{'='*60}\n", 'info')

        self.progress_label.config(text=f"Running Step {step_num}...")
        self.progress_var.set(0)

        if step_num in self.step_frames:
            self.step_frames[step_num]['status'].config(text="Running", bg=self.colors['accent'])
            self.step_frames[step_num]['run_btn'].config(state=tk.DISABLED)

        def run_script():
            try:
                config_file = BASE_DIR / "pipeline_config.json"
                config = {}
                if config_file.exists():
                    with open(config_file, 'r') as f:
                        config = json.load(f)

                cmd = [sys.executable, str(script_path)]

                # Special handling for different script types
                if step_num == 3:  # Well integration uses -c config or LAS directory
                    module_config = script_path.parent / "config_default.json"
                    if module_config.exists():
                        cmd.extend(["-c", str(module_config)])
                    elif CONFIG_AVAILABLE:
                        # Get LAS directory from centralized config
                        proj_config = get_config()
                        if proj_config.well_logs_directory:
                            cmd.append(proj_config.well_logs_directory)
                            output_dir = proj_config.output_directory or str(BASE_DIR / "well_outputs")
                            cmd.extend(["-o", str(Path(output_dir) / "well_outputs")])
                    else:
                        # Fallback to output directory relative path
                        cmd.extend(["-o", str(BASE_DIR / "well_outputs")])
                elif step_num == 9:  # 2D-3D Integration uses -c CONFIG format
                    module_config = script_path.parent / "config_default.json"
                    if module_config.exists():
                        cmd.extend(["-c", str(module_config)])
                elif step_num == 10:  # REAL Interpretation runs standalone
                    pass  # Uses hardcoded paths internally
                elif step_num in [5, 6, 7, 8]:  # Interpretation scripts take no args
                    pass  # These scripts use hardcoded paths internally
                else:
                    # Standard scripts: positional SEGY + --output-dir
                    if config.get("input_segy_3d"):
                        cmd.append(config["input_segy_3d"])
                    output_dir = config.get("output_dir", str(BASE_DIR / "outputs"))
                    cmd.extend(["--output-dir", output_dir])

                self.current_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    cwd=str(script_path.parent)
                )

                for line in iter(self.current_process.stdout.readline, ''):
                    self.process_queue.put(('output', line.rstrip()))

                    # Parse progress if available
                    if '%' in line:
                        try:
                            pct = float(re.search(r'(\d+)%', line).group(1))
                            self.process_queue.put(('progress', pct))
                        except:
                            pass

                self.current_process.wait()

                if self.current_process.returncode == 0:
                    self.process_queue.put(('complete', step_num))
                else:
                    self.process_queue.put(('error', f"Step {step_num} failed"))

            except Exception as e:
                self.process_queue.put(('error', str(e)))

        threading.Thread(target=run_script, daemon=True).start()

    def _process_queue(self):
        try:
            while True:
                msg_type, content = self.process_queue.get_nowait()

                if msg_type == 'output':
                    tag = None
                    if 'error' in content.lower():
                        tag = 'error'
                    elif 'warning' in content.lower():
                        tag = 'warning'
                    elif 'success' in content.lower() or 'complete' in content.lower():
                        tag = 'success'
                    self.terminal_print(content, tag)

                elif msg_type == 'progress':
                    self.progress_var.set(content)

                elif msg_type == 'complete':
                    step_num = content
                    self.state.mark_completed(step_num)
                    self.terminal_print(f"\nStep {step_num} completed successfully!", 'success')
                    self.progress_label.config(text="Complete")
                    self.progress_var.set(100)

                    if step_num in self.step_frames:
                        self.step_frames[step_num]['status'].config(text="Done", bg=self.colors['success'])
                        self.step_frames[step_num]['run_btn'].config(state=tk.NORMAL)
                    if step_num in self.status_labels:
                        self.status_labels[step_num].config(text="Done", bg=self.colors['success'], fg='white')

                    completed = len(self.state.state['completed_steps'])
                    self.progress_status.config(text=f"Steps: {completed}/8 completed")

                elif msg_type == 'error':
                    self.terminal_print(f"Error: {content}", 'error')
                    self.progress_label.config(text="Error")

        except queue.Empty:
            pass

        self.root.after(100, self._process_queue)

    def stop_processing(self):
        if self.current_process:
            self.current_process.terminate()
            self.terminal_print("Processing stopped by user", 'warning')
            self.progress_label.config(text="Stopped")

    def run_all_steps(self):
        pending = [i for i in range(1, 9) if not self.state.is_completed(i)]
        if not pending:
            messagebox.showinfo("Complete", "All steps are already completed!")
            return

        if messagebox.askyesno("Run All", f"Run pending steps: {pending}?"):
            for step_num in pending:
                self.run_processing_step(step_num)

    def load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp")]
        )
        if path:
            self._display_image(path)

    def load_output_image(self):
        outputs_figures = BASE_DIR / "outputs" / "figures"
        if outputs_figures.exists():
            initial_dir = outputs_figures
        else:
            initial_dir = BASE_DIR / "outputs"

        path = filedialog.askopenfilename(
            initialdir=initial_dir,
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp")]
        )
        if path:
            self._display_image(path)

    def _display_image(self, path: str):
        self.current_image_path = path
        try:
            # Try to display image
            from PIL import Image, ImageTk
            img = Image.open(path)
            img.thumbnail((450, 400))
            photo = ImageTk.PhotoImage(img)
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo
        except ImportError:
            self.image_label.config(text=f"Image loaded:\n{Path(path).name}\n\n(Install Pillow for preview:\npip install Pillow)")
        except Exception as e:
            self.image_label.config(text=f"Error loading image:\n{e}")

    def interpret_current_image(self):
        if not self.current_image_path:
            messagebox.showwarning("No Image", "Please load an image first.")
            return

        if not self.ollama.has_vision:
            messagebox.showwarning("No Vision Model", "Llava vision model not available.\nRun: ollama pull llava:13b")
            return

        self.image_interpretation.delete(1.0, tk.END)
        self.image_interpretation.insert(tk.END, "Analyzing image...")

        additional = self.image_prompt.get()

        def interpret():
            result = self.interpreter.interpret_image(self.current_image_path, additional)
            self.root.after(0, lambda: self._show_image_result(result))

        threading.Thread(target=interpret, daemon=True).start()

    def _show_image_result(self, result: str):
        self.image_interpretation.delete(1.0, tk.END)
        self.image_interpretation.insert(tk.END, result)

    def generate_report(self, report_type: str):
        self.add_chat_message("system", f"Generating {report_type} report...")
        self.status_label.config(text="Generating report...")

        def generate():
            result = self.interpreter.generate_report(report_type)
            self.root.after(0, lambda: self._show_report(result, report_type))

        threading.Thread(target=generate, daemon=True).start()

    def _show_report(self, content: str, report_type: str):
        self.add_chat_message("ai", content)
        self.status_label.config(text="Ready")

        # Offer to save
        if messagebox.askyesno("Save Report", "Would you like to save this report?"):
            self.save_report(content, report_type)

    def save_report(self, content: str, report_type: str):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"seismic_report_{report_type}_{timestamp}.md"

        path = filedialog.asksaveasfilename(
            defaultextension=".md",
            filetypes=[("Markdown", "*.md"), ("Text", "*.txt")],
            initialfile=default_name
        )
        if path:
            with open(path, 'w') as f:
                f.write(f"# Seismic Interpretation Report - {report_type.title()}\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(content)
            messagebox.showinfo("Saved", f"Report saved to:\n{path}")

    def export_report(self):
        self.generate_report("full")

    def export_chat(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt")]
        )
        if path:
            content = self.chat_display.get("1.0", tk.END)
            with open(path, 'w') as f:
                f.write(content)
            messagebox.showinfo("Exported", f"Chat exported to {path}")

    def new_project(self):
        if messagebox.askyesno("New Project", "Start a new project? This will reset all progress."):
            self.state.reset_all()
            for entry in self.data_entries.values():
                entry.delete(0, tk.END)
            self.clear_terminal()
            self.chat_display.config(state=tk.NORMAL)
            self.chat_display.delete(1.0, tk.END)
            self.chat_display.config(state=tk.DISABLED)
            self._show_welcome()

    def open_project(self):
        path = filedialog.askdirectory(title="Select Project Folder")
        if path:
            # Look for pipeline_config.json
            config_file = Path(path) / "pipeline_config.json"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
                # Update entries
                if "input_segy_3d" in config:
                    self.data_entries["segy_3d"].delete(0, tk.END)
                    self.data_entries["segy_3d"].insert(0, config["input_segy_3d"])
                messagebox.showinfo("Loaded", "Project configuration loaded!")

    def show_about(self):
        messagebox.showinfo(
            "About",
            "Seismic AI Assistant v2.0\n\n"
            "PhD Research - Bornu Chad Basin\n"
            "Author: Moses Ekene Obasi\n"
            "University of Calabar, Nigeria\n\n"
            "Powered by Ollama LLM\n"
            "Vision: Llava 13B"
        )


# =============================================================================
# MAIN
# =============================================================================

def main():
    root = tk.Tk()

    # Style
    style = ttk.Style()
    style.theme_use('clam')

    app = SeismicAIAssistantGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
