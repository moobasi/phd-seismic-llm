"""
================================================================================
2D SEISMIC PROCESSING AND INTEGRATION MODULE v1.0
PhD Research - Regional Framework and 2D/3D Integration
================================================================================

Author: Moses Ekene Obasi
Institution: University of Calabar, Nigeria
Supervisor: Prof. Dominic Akam Obi

Features:
- Multi-line 2D seismic inventory and QC
- Automatic line geometry extraction
- Well-to-2D seismic tie
- 2D/3D integration and comparison
- Regional horizon correlation
- Composite section generation
- Publication-ready figures

Usage:
    python seismic_2d_automation.py "path/to/2d_folder" -o "outputs"
    python seismic_2d_automation.py -c config.json
    python seismic_2d_automation.py --inventory  # Quick inventory scan
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import argparse
import logging
import json
import os
import warnings

warnings.filterwarnings('ignore')

# Try importing segyio
try:
    import segyio
    SEGYIO_AVAILABLE = True
except ImportError:
    SEGYIO_AVAILABLE = False
    print("WARNING: segyio not installed. Install with: pip install segyio")

# Try importing pandas for well header
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Seismic2DConfig:
    """Configuration for 2D seismic processing"""

    # Input/Output
    input_directory: str = ""
    output_dir: str = "2d_outputs"

    # 3D reference (for integration)
    reference_3d_segy: str = ""

    # Well data
    well_header_file: str = ""  # Excel/CSV with well locations
    las_directory: str = ""

    # Line filtering
    line_prefix_filter: List[str] = field(default_factory=lambda: ["CH-78"])
    include_migrated: bool = True
    include_stacked: bool = True

    # Processing options
    apply_agc: bool = True
    agc_window_ms: float = 500.0
    normalize_amplitudes: bool = True

    # QC thresholds
    min_traces: int = 100
    max_dead_trace_pct: float = 10.0

    # Visualization
    save_figures: bool = True
    figure_dpi: int = 300
    colormap: str = "seismic"

    # Output formats
    export_inventory_csv: bool = True
    export_line_images: bool = True
    create_basemap: bool = True

    @classmethod
    def from_json(cls, path: str) -> 'Seismic2DConfig':
        """Load configuration from JSON file, ignoring unknown fields"""
        with open(path, 'r') as f:
            data = json.load(f)
        # Only use fields that exist in this dataclass
        valid_fields = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid_fields)

    def to_json(self, path: str):
        """Save configuration to JSON file"""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)


@dataclass
class Line2DInfo:
    """Information about a single 2D seismic line"""
    filename: str
    filepath: str
    line_name: str
    survey_id: str
    processing_type: str  # MIG, STK, etc.

    # Geometry
    n_traces: int = 0
    n_samples: int = 0
    sample_interval_ms: float = 0.0
    record_length_ms: float = 0.0

    # Coordinates (if available)
    start_x: Optional[float] = None
    start_y: Optional[float] = None
    end_x: Optional[float] = None
    end_y: Optional[float] = None
    line_length_m: Optional[float] = None

    # CDP range
    cdp_start: Optional[int] = None
    cdp_end: Optional[int] = None

    # Quality metrics
    dead_trace_count: int = 0
    dead_trace_pct: float = 0.0
    amplitude_range: Tuple[float, float] = (0.0, 0.0)
    rms_amplitude: float = 0.0

    # Status
    status: str = "PENDING"
    quality_tier: str = "UNKNOWN"
    error_message: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Seismic2DResults:
    """Results from 2D seismic processing"""

    timestamp: str = ""
    version: str = "1.0"
    processing_time_seconds: float = 0.0
    input_directory: str = ""

    # Inventory
    total_files_found: int = 0
    lines_processed: int = 0
    lines_failed: int = 0

    # Line inventory
    line_inventory: List[Dict[str, Any]] = field(default_factory=list)

    # Survey summary
    survey_summary: Dict[str, Any] = field(default_factory=dict)

    # Quality summary
    quality_summary: Dict[str, Any] = field(default_factory=dict)

    # Well ties
    well_ties: List[Dict[str, Any]] = field(default_factory=list)

    # Output files
    output_files: Dict[str, str] = field(default_factory=dict)

    # Warnings and recommendations
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


# =============================================================================
# LINE PARSER
# =============================================================================

class LineNameParser:
    """Parse 2D line names from filenames"""

    @staticmethod
    def parse(filename: str) -> Dict[str, str]:
        """
        Parse filename to extract line information.

        Expected formats:
        - CH-78-102_MIG_B1410.SGY
        - CH-78-102_STK_B0196.SGY
        - CH-78-400S. EXT_STK_B0214.SGY
        """
        result = {
            "survey_id": "",
            "line_name": "",
            "processing_type": "",
            "block_id": "",
            "raw_name": filename
        }

        name = Path(filename).stem.upper()

        # Try to extract survey ID (e.g., CH-78-102)
        if "CH-78" in name or "CH78" in name:
            parts = name.replace("CH78", "CH-78").split("_")

            if len(parts) >= 1:
                # First part is usually survey-line (e.g., CH-78-102)
                result["survey_id"] = parts[0].replace(".", "").strip()
                result["line_name"] = parts[0]

            # Look for processing type
            for part in parts:
                if part in ["MIG", "STK", "PSTM", "PSDM", "DMO"]:
                    result["processing_type"] = part
                elif part.startswith("B") and part[1:].isdigit():
                    result["block_id"] = part
        else:
            # Generic parsing
            result["line_name"] = name
            if "_MIG" in name or "_MIGR" in name:
                result["processing_type"] = "MIG"
            elif "_STK" in name or "_STACK" in name:
                result["processing_type"] = "STK"

        return result


# =============================================================================
# MAIN AUTOMATION CLASS
# =============================================================================

class Seismic2DAutomation:
    """
    2D Seismic Processing and Integration Automation
    """

    def __init__(self, config: Seismic2DConfig):
        self.config = config
        self.results = Seismic2DResults()
        self.parser = LineNameParser()

        # Storage
        self.line_inventory: List[Line2DInfo] = []
        self.well_locations: Dict[str, Dict] = {}

        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # INVENTORY SCANNING
    # =========================================================================

    def scan_directory(self) -> List[str]:
        """Scan directory for SEGY files"""
        logger.info(f"Scanning directory: {self.config.input_directory}")

        input_dir = Path(self.config.input_directory)
        segy_files = []

        # Find all SEGY files
        for ext in ["*.sgy", "*.segy", "*.SGY", "*.SEGY"]:
            segy_files.extend(input_dir.glob(ext))
            # Also check subdirectories
            segy_files.extend(input_dir.glob(f"**/{ext}"))

        # Remove duplicates and sort
        segy_files = sorted(set(segy_files))

        # Apply filters
        filtered_files = []
        for f in segy_files:
            name = f.name.upper()

            # Check prefix filter
            if self.config.line_prefix_filter:
                if not any(prefix.upper() in name for prefix in self.config.line_prefix_filter):
                    continue

            # Check processing type filter
            if not self.config.include_migrated and "_MIG" in name:
                continue
            if not self.config.include_stacked and "_STK" in name:
                continue

            filtered_files.append(str(f))

        logger.info(f"Found {len(segy_files)} SEGY files, {len(filtered_files)} after filtering")

        self.results.total_files_found = len(filtered_files)
        return filtered_files

    def build_inventory(self, quick_scan: bool = False) -> List[Line2DInfo]:
        """Build inventory of all 2D lines"""
        logger.info("Building 2D line inventory...")

        segy_files = self.scan_directory()

        if not segy_files:
            logger.warning("No SEGY files found!")
            return []

        inventory = []

        for filepath in tqdm(segy_files, desc="Scanning lines"):
            line_info = self._analyze_line(filepath, quick_scan=quick_scan)
            inventory.append(line_info)

        self.line_inventory = inventory
        self.results.line_inventory = [l.to_dict() for l in inventory]
        self.results.lines_processed = sum(1 for l in inventory if l.status == "OK")
        self.results.lines_failed = sum(1 for l in inventory if l.status == "ERROR")

        # Generate summary
        self._generate_survey_summary()

        return inventory

    def _analyze_line(self, filepath: str, quick_scan: bool = False) -> Line2DInfo:
        """Analyze a single 2D line"""
        filename = Path(filepath).name
        parsed = self.parser.parse(filename)

        line_info = Line2DInfo(
            filename=filename,
            filepath=filepath,
            line_name=parsed["line_name"],
            survey_id=parsed["survey_id"],
            processing_type=parsed["processing_type"]
        )

        if not SEGYIO_AVAILABLE:
            line_info.status = "SKIPPED"
            line_info.error_message = "segyio not available"
            return line_info

        try:
            with segyio.open(filepath, "r", ignore_geometry=True) as f:
                # Basic geometry
                line_info.n_traces = f.tracecount
                line_info.n_samples = len(f.samples)
                line_info.sample_interval_ms = segyio.tools.dt(f) / 1000.0
                line_info.record_length_ms = line_info.n_samples * line_info.sample_interval_ms

                # Try to get coordinates from headers
                try:
                    first_trace = f.header[0]
                    last_trace = f.header[-1]

                    # Scaled coordinates
                    scalar = first_trace.get(segyio.TraceField.SourceGroupScalar, 1)
                    if scalar < 0:
                        scalar = -1.0 / scalar
                    elif scalar == 0:
                        scalar = 1

                    line_info.start_x = first_trace.get(segyio.TraceField.SourceX, 0) * scalar
                    line_info.start_y = first_trace.get(segyio.TraceField.SourceY, 0) * scalar
                    line_info.end_x = last_trace.get(segyio.TraceField.SourceX, 0) * scalar
                    line_info.end_y = last_trace.get(segyio.TraceField.SourceY, 0) * scalar

                    # CDP range
                    line_info.cdp_start = first_trace.get(segyio.TraceField.CDP, 0)
                    line_info.cdp_end = last_trace.get(segyio.TraceField.CDP, 0)

                    # Calculate line length
                    if line_info.start_x and line_info.end_x:
                        dx = line_info.end_x - line_info.start_x
                        dy = line_info.end_y - line_info.start_y
                        line_info.line_length_m = np.sqrt(dx**2 + dy**2)
                except:
                    pass

                if not quick_scan:
                    # QC metrics - sample traces
                    n_sample = min(100, line_info.n_traces)
                    sample_indices = np.linspace(0, line_info.n_traces - 1, n_sample, dtype=int)

                    amplitudes = []
                    dead_count = 0

                    for idx in sample_indices:
                        trace = f.trace[idx]
                        rms = np.sqrt(np.mean(trace**2))

                        if rms < 1e-10:
                            dead_count += 1
                        amplitudes.append(rms)

                    line_info.dead_trace_pct = (dead_count / n_sample) * 100
                    line_info.dead_trace_count = int(line_info.dead_trace_pct * line_info.n_traces / 100)

                    if amplitudes:
                        all_amps = np.array(amplitudes)
                        line_info.rms_amplitude = float(np.mean(all_amps))
                        line_info.amplitude_range = (float(np.min(all_amps)), float(np.max(all_amps)))

                # Assign quality tier
                if line_info.n_traces < self.config.min_traces:
                    line_info.quality_tier = "EXCLUDED"
                elif line_info.dead_trace_pct > self.config.max_dead_trace_pct:
                    line_info.quality_tier = "POOR"
                elif line_info.dead_trace_pct > 5.0:
                    line_info.quality_tier = "FAIR"
                else:
                    line_info.quality_tier = "GOOD"

                line_info.status = "OK"

        except Exception as e:
            line_info.status = "ERROR"
            line_info.error_message = str(e)
            logger.warning(f"Error analyzing {filename}: {e}")

        return line_info

    def _generate_survey_summary(self):
        """Generate summary statistics by survey"""
        surveys = {}

        for line in self.line_inventory:
            survey_id = line.survey_id or "UNKNOWN"

            if survey_id not in surveys:
                surveys[survey_id] = {
                    "line_count": 0,
                    "total_traces": 0,
                    "processing_types": set(),
                    "quality_good": 0,
                    "quality_fair": 0,
                    "quality_poor": 0
                }

            surveys[survey_id]["line_count"] += 1
            surveys[survey_id]["total_traces"] += line.n_traces

            if line.processing_type:
                surveys[survey_id]["processing_types"].add(line.processing_type)

            if line.quality_tier == "GOOD":
                surveys[survey_id]["quality_good"] += 1
            elif line.quality_tier == "FAIR":
                surveys[survey_id]["quality_fair"] += 1
            elif line.quality_tier == "POOR":
                surveys[survey_id]["quality_poor"] += 1

        # Convert sets to lists for JSON
        for survey_id in surveys:
            surveys[survey_id]["processing_types"] = list(surveys[survey_id]["processing_types"])

        self.results.survey_summary = surveys

        # Quality summary
        quality_counts = {}
        for line in self.line_inventory:
            tier = line.quality_tier
            quality_counts[tier] = quality_counts.get(tier, 0) + 1

        self.results.quality_summary = {
            "tier_counts": quality_counts,
            "total_lines": len(self.line_inventory),
            "usable_lines": sum(1 for l in self.line_inventory if l.quality_tier in ["GOOD", "FAIR"])
        }

    # =========================================================================
    # WELL DATA LOADING
    # =========================================================================

    def load_well_locations(self) -> Dict[str, Dict]:
        """Load well locations from header file"""
        if not self.config.well_header_file:
            logger.info("No well header file specified")
            return {}

        if not PANDAS_AVAILABLE:
            logger.warning("pandas not available for reading Excel files")
            return {}

        header_path = Path(self.config.well_header_file)
        if not header_path.exists():
            logger.warning(f"Well header file not found: {header_path}")
            return {}

        logger.info(f"Loading well locations from: {header_path}")

        try:
            if header_path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(header_path)
            else:
                df = pd.read_csv(header_path)

            # Try to identify coordinate columns
            x_cols = [c for c in df.columns if any(x in c.upper() for x in ['EASTING', 'X_COORD', 'X', 'LONG'])]
            y_cols = [c for c in df.columns if any(x in c.upper() for x in ['NORTHING', 'Y_COORD', 'Y', 'LAT'])]
            name_cols = [c for c in df.columns if any(x in c.upper() for x in ['WELL', 'NAME', 'WELLNAME'])]

            if not (x_cols and y_cols and name_cols):
                logger.warning("Could not identify coordinate columns in well header")
                # Try first few columns
                if len(df.columns) >= 3:
                    name_cols = [df.columns[0]]
                    x_cols = [df.columns[1]]
                    y_cols = [df.columns[2]]

            wells = {}
            for _, row in df.iterrows():
                well_name = str(row[name_cols[0]]).strip().upper()
                x = float(row[x_cols[0]]) if pd.notna(row[x_cols[0]]) else None
                y = float(row[y_cols[0]]) if pd.notna(row[y_cols[0]]) else None

                if well_name and x and y:
                    wells[well_name] = {
                        "name": well_name,
                        "x": x,
                        "y": y,
                        "easting": x,
                        "northing": y
                    }

            logger.info(f"Loaded {len(wells)} well locations")
            self.well_locations = wells
            return wells

        except Exception as e:
            logger.error(f"Error loading well header: {e}")
            return {}

    # =========================================================================
    # WELL-TO-SEISMIC TIE
    # =========================================================================

    def find_nearest_lines(self, well_name: str, max_distance_m: float = 5000) -> List[Dict]:
        """Find 2D lines nearest to a well"""
        if well_name not in self.well_locations:
            logger.warning(f"Well {well_name} not found in locations")
            return []

        well = self.well_locations[well_name]
        well_x, well_y = well["x"], well["y"]

        nearest = []

        for line in self.line_inventory:
            if line.status != "OK":
                continue

            if line.start_x is None or line.end_x is None:
                continue

            # Calculate perpendicular distance from well to line
            dist = self._point_to_line_distance(
                well_x, well_y,
                line.start_x, line.start_y,
                line.end_x, line.end_y
            )

            if dist is not None and dist <= max_distance_m:
                nearest.append({
                    "line_name": line.line_name,
                    "filename": line.filename,
                    "distance_m": dist,
                    "processing_type": line.processing_type,
                    "quality_tier": line.quality_tier
                })

        # Sort by distance
        nearest.sort(key=lambda x: x["distance_m"])

        return nearest

    def _point_to_line_distance(self, px: float, py: float,
                                 x1: float, y1: float,
                                 x2: float, y2: float) -> Optional[float]:
        """Calculate perpendicular distance from point to line segment"""
        try:
            # Line vector
            dx = x2 - x1
            dy = y2 - y1

            # Length squared
            l2 = dx*dx + dy*dy
            if l2 == 0:
                return np.sqrt((px - x1)**2 + (py - y1)**2)

            # Projection parameter
            t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / l2))

            # Closest point on line
            proj_x = x1 + t * dx
            proj_y = y1 + t * dy

            return np.sqrt((px - proj_x)**2 + (py - proj_y)**2)
        except:
            return None

    def analyze_well_ties(self) -> List[Dict]:
        """Analyze potential well-to-2D ties for all wells"""
        logger.info("Analyzing well-to-2D ties...")

        if not self.well_locations:
            self.load_well_locations()

        if not self.well_locations:
            logger.warning("No well locations available")
            return []

        ties = []

        for well_name, well_info in self.well_locations.items():
            nearest_lines = self.find_nearest_lines(well_name)

            if nearest_lines:
                tie_info = {
                    "well_name": well_name,
                    "well_x": well_info["x"],
                    "well_y": well_info["y"],
                    "nearest_lines": nearest_lines[:5],  # Top 5
                    "n_lines_within_5km": len(nearest_lines)
                }
                ties.append(tie_info)

        self.results.well_ties = ties

        logger.info(f"Found potential ties for {len(ties)} wells")
        return ties

    # =========================================================================
    # VISUALIZATION
    # =========================================================================

    def create_basemap(self):
        """Create basemap showing 2D lines and wells"""
        logger.info("Creating basemap...")

        fig, ax = plt.subplots(figsize=(14, 12))

        # Plot 2D lines
        lines_with_coords = [l for l in self.line_inventory
                            if l.start_x is not None and l.status == "OK"]

        if not lines_with_coords:
            logger.warning("No lines with coordinates for basemap")
            plt.close()
            return

        # Color by quality
        quality_colors = {
            "GOOD": "green",
            "FAIR": "orange",
            "POOR": "red",
            "EXCLUDED": "gray"
        }

        for line in lines_with_coords:
            color = quality_colors.get(line.quality_tier, "blue")
            ax.plot([line.start_x, line.end_x], [line.start_y, line.end_y],
                   color=color, linewidth=1.5, alpha=0.7)

            # Add line label at midpoint
            mid_x = (line.start_x + line.end_x) / 2
            mid_y = (line.start_y + line.end_y) / 2
            ax.annotate(line.line_name, (mid_x, mid_y), fontsize=6, alpha=0.7)

        # Plot wells
        if self.well_locations:
            well_x = [w["x"] for w in self.well_locations.values()]
            well_y = [w["y"] for w in self.well_locations.values()]
            well_names = list(self.well_locations.keys())

            ax.scatter(well_x, well_y, c='red', s=100, marker='^',
                      edgecolors='black', linewidths=1, zorder=10, label='Wells')

            for name, x, y in zip(well_names, well_x, well_y):
                ax.annotate(name, (x, y), xytext=(5, 5), textcoords='offset points',
                           fontsize=8, fontweight='bold')

        # Legend
        legend_patches = [
            mpatches.Patch(color='green', label='Good Quality'),
            mpatches.Patch(color='orange', label='Fair Quality'),
            mpatches.Patch(color='red', label='Poor Quality'),
        ]
        ax.legend(handles=legend_patches, loc='upper right')

        ax.set_xlabel('Easting (m)', fontweight='bold')
        ax.set_ylabel('Northing (m)', fontweight='bold')
        ax.set_title('2D Seismic Line Basemap - Bornu Chad Basin', fontweight='bold', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        plt.tight_layout()

        # Save
        fig_path = Path(self.config.output_dir) / "basemap_2d_lines.png"
        fig.savefig(fig_path, dpi=self.config.figure_dpi, bbox_inches='tight')
        plt.close()

        self.results.output_files['basemap'] = str(fig_path)
        logger.info(f"Basemap saved to {fig_path}")

    def create_inventory_plot(self):
        """Create inventory summary plots"""
        logger.info("Creating inventory plots...")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Quality distribution
        ax1 = axes[0, 0]
        quality_counts = {}
        for line in self.line_inventory:
            tier = line.quality_tier
            quality_counts[tier] = quality_counts.get(tier, 0) + 1

        colors = ['green', 'orange', 'red', 'gray']
        tiers = ['GOOD', 'FAIR', 'POOR', 'EXCLUDED']
        counts = [quality_counts.get(t, 0) for t in tiers]

        ax1.bar(tiers, counts, color=colors)
        ax1.set_xlabel('Quality Tier')
        ax1.set_ylabel('Number of Lines')
        ax1.set_title('Line Quality Distribution')

        # Add count labels
        for i, (tier, count) in enumerate(zip(tiers, counts)):
            ax1.text(i, count + 0.5, str(count), ha='center', fontweight='bold')

        # 2. Processing type distribution
        ax2 = axes[0, 1]
        proc_counts = {}
        for line in self.line_inventory:
            proc = line.processing_type or "UNKNOWN"
            proc_counts[proc] = proc_counts.get(proc, 0) + 1

        ax2.bar(proc_counts.keys(), proc_counts.values(), color='steelblue')
        ax2.set_xlabel('Processing Type')
        ax2.set_ylabel('Number of Lines')
        ax2.set_title('Processing Type Distribution')

        # 3. Line length histogram
        ax3 = axes[1, 0]
        lengths = [l.line_length_m / 1000 for l in self.line_inventory
                  if l.line_length_m is not None and l.line_length_m > 0]

        if lengths:
            ax3.hist(lengths, bins=20, color='teal', edgecolor='black', alpha=0.7)
            ax3.set_xlabel('Line Length (km)')
            ax3.set_ylabel('Count')
            ax3.set_title('Line Length Distribution')
            ax3.axvline(np.median(lengths), color='red', linestyle='--',
                       label=f'Median: {np.median(lengths):.1f} km')
            ax3.legend()

        # 4. Dead trace percentage
        ax4 = axes[1, 1]
        dead_pcts = [l.dead_trace_pct for l in self.line_inventory if l.status == "OK"]

        if dead_pcts:
            ax4.hist(dead_pcts, bins=20, color='coral', edgecolor='black', alpha=0.7)
            ax4.set_xlabel('Dead Trace Percentage')
            ax4.set_ylabel('Count')
            ax4.set_title('Data Quality (Dead Traces)')
            ax4.axvline(self.config.max_dead_trace_pct, color='red', linestyle='--',
                       label=f'Threshold: {self.config.max_dead_trace_pct}%')
            ax4.legend()

        plt.suptitle('2D Seismic Line Inventory Summary', fontweight='bold', fontsize=14)
        plt.tight_layout()

        fig_path = Path(self.config.output_dir) / "inventory_summary.png"
        fig.savefig(fig_path, dpi=self.config.figure_dpi, bbox_inches='tight')
        plt.close()

        self.results.output_files['inventory_plot'] = str(fig_path)
        logger.info(f"Inventory plot saved to {fig_path}")

    def create_line_display(self, line_name: str, apply_agc: bool = True):
        """Create display of a single 2D line"""
        # Find line
        line = next((l for l in self.line_inventory if l.line_name == line_name), None)

        if not line:
            logger.warning(f"Line {line_name} not found")
            return

        if not SEGYIO_AVAILABLE:
            logger.warning("segyio not available")
            return

        logger.info(f"Creating display for line: {line_name}")

        try:
            with segyio.open(line.filepath, "r", ignore_geometry=True) as f:
                # Read data
                data = segyio.tools.collect(f.trace[:])

                # Apply AGC if requested
                if apply_agc and self.config.apply_agc:
                    data = self._apply_agc(data, f.samples, self.config.agc_window_ms)

                # Normalize
                if self.config.normalize_amplitudes:
                    data = data / (np.percentile(np.abs(data), 99) + 1e-10)

                # Create figure
                fig, ax = plt.subplots(figsize=(16, 10))

                extent = [0, data.shape[0], f.samples[-1], f.samples[0]]

                vmax = np.percentile(np.abs(data), 98)

                im = ax.imshow(data.T, aspect='auto', cmap=self.config.colormap,
                              extent=extent, vmin=-vmax, vmax=vmax)

                ax.set_xlabel('Trace Number', fontweight='bold')
                ax.set_ylabel('Time (ms)', fontweight='bold')
                ax.set_title(f'2D Seismic Line: {line_name}\n'
                           f'Processing: {line.processing_type} | '
                           f'Traces: {line.n_traces} | '
                           f'Quality: {line.quality_tier}',
                           fontweight='bold')

                plt.colorbar(im, ax=ax, label='Amplitude', shrink=0.8)

                plt.tight_layout()

                # Save
                safe_name = line_name.replace("/", "_").replace("\\", "_").replace(" ", "_")
                fig_path = Path(self.config.output_dir) / f"line_{safe_name}.png"
                fig.savefig(fig_path, dpi=150, bbox_inches='tight')
                plt.close()

                logger.info(f"Line display saved to {fig_path}")
                return str(fig_path)

        except Exception as e:
            logger.error(f"Error creating line display: {e}")
            return None

    def _apply_agc(self, data: np.ndarray, samples: np.ndarray,
                   window_ms: float) -> np.ndarray:
        """Apply Automatic Gain Control"""
        sample_interval = samples[1] - samples[0] if len(samples) > 1 else 4.0
        window_samples = int(window_ms / sample_interval)

        if window_samples < 3:
            window_samples = 3

        agc_data = np.zeros_like(data)

        for i in range(data.shape[0]):
            trace = data[i, :]

            # Calculate envelope using running RMS
            envelope = np.zeros_like(trace)
            half_win = window_samples // 2

            for j in range(len(trace)):
                start = max(0, j - half_win)
                end = min(len(trace), j + half_win)
                envelope[j] = np.sqrt(np.mean(trace[start:end]**2)) + 1e-10

            agc_data[i, :] = trace / envelope

        return agc_data

    # =========================================================================
    # COMPREHENSIVE EDA (PhD-LEVEL ANALYSIS)
    # =========================================================================

    def run_comprehensive_eda(self, n_representative_lines: int = 5) -> Dict[str, Any]:
        """
        Run comprehensive Exploratory Data Analysis on 2D lines.
        This provides PhD-level documentation of data quality and characteristics.
        """
        logger.info("Running comprehensive 2D EDA...")

        eda_results = {
            'processing_state_analysis': {},
            'frequency_analysis': {},
            'amplitude_analysis': {},
            'representative_lines': [],
            'mig_vs_stk_comparison': {},
            'recommendations': []
        }

        # 1. Processing State Analysis
        logger.info("  Analyzing processing states...")
        eda_results['processing_state_analysis'] = self._analyze_processing_states()

        # 2. Select representative lines for detailed analysis
        logger.info("  Selecting representative lines...")
        rep_lines = self._select_representative_lines(n_representative_lines)
        eda_results['representative_lines'] = [l.line_name for l in rep_lines]

        # 3. Detailed analysis on representative lines
        logger.info("  Running detailed spectral analysis...")
        eda_results['frequency_analysis'] = self._analyze_frequency_content(rep_lines)

        # 4. Amplitude analysis
        logger.info("  Running amplitude analysis...")
        eda_results['amplitude_analysis'] = self._analyze_amplitude_characteristics(rep_lines)

        # 5. MIG vs STK comparison
        logger.info("  Comparing MIG vs STK processing...")
        eda_results['mig_vs_stk_comparison'] = self._compare_mig_stk()

        # 6. Generate EDA recommendations
        eda_results['recommendations'] = self._generate_eda_recommendations(eda_results)

        # 7. Create EDA figures
        if self.config.save_figures:
            logger.info("  Creating EDA figures...")
            self._create_eda_figures(eda_results, rep_lines)

        # Store in results
        self.results.quality_summary['eda'] = eda_results

        return eda_results

    def _analyze_processing_states(self) -> Dict[str, Any]:
        """Analyze processing states across all lines"""
        states = {
            'MIG': {'count': 0, 'lines': []},
            'STK': {'count': 0, 'lines': []},
            'PSTM': {'count': 0, 'lines': []},
            'PSDM': {'count': 0, 'lines': []},
            'UNKNOWN': {'count': 0, 'lines': []}
        }

        for line in self.line_inventory:
            proc = line.processing_type if line.processing_type else 'UNKNOWN'
            if proc not in states:
                states[proc] = {'count': 0, 'lines': []}
            states[proc]['count'] += 1
            states[proc]['lines'].append(line.line_name)

        # Summary
        total = len(self.line_inventory)
        summary = {
            'total_lines': total,
            'processing_breakdown': {k: v['count'] for k, v in states.items() if v['count'] > 0},
            'recommended_for_interpretation': 'MIG' if states['MIG']['count'] > 0 else 'STK',
            'mig_available': states['MIG']['count'] > 0,
            'stk_available': states['STK']['count'] > 0
        }

        return summary

    def _select_representative_lines(self, n: int) -> List[Line2DInfo]:
        """Select representative lines for detailed analysis"""
        # Prefer MIG over STK, good quality over poor
        mig_lines = [l for l in self.line_inventory
                     if l.processing_type == 'MIG' and l.quality_tier == 'GOOD' and l.status == 'OK']
        stk_lines = [l for l in self.line_inventory
                     if l.processing_type == 'STK' and l.quality_tier == 'GOOD' and l.status == 'OK']
        other_lines = [l for l in self.line_inventory
                       if l.quality_tier in ['GOOD', 'FAIR'] and l.status == 'OK']

        # Prioritize: MIG > STK > other
        candidates = mig_lines + stk_lines + other_lines

        # Remove duplicates while preserving order
        seen = set()
        unique = []
        for line in candidates:
            if line.line_name not in seen:
                seen.add(line.line_name)
                unique.append(line)

        # Select evenly distributed by line length
        if len(unique) <= n:
            return unique

        unique.sort(key=lambda x: x.line_length_m or 0)
        step = len(unique) // n
        return [unique[i * step] for i in range(n)]

    def _analyze_frequency_content(self, lines: List[Line2DInfo]) -> Dict[str, Any]:
        """Analyze frequency content of representative lines"""
        if not SEGYIO_AVAILABLE:
            return {'error': 'segyio not available'}

        freq_results = {
            'per_line': [],
            'summary': {}
        }

        all_dominant_freqs = []
        all_bandwidths = []

        for line in lines:
            try:
                with segyio.open(line.filepath, "r", ignore_geometry=True) as f:
                    # Sample traces for spectral analysis
                    n_traces = f.tracecount
                    n_sample = min(50, n_traces)
                    sample_indices = np.linspace(0, n_traces - 1, n_sample, dtype=int)

                    dt_ms = segyio.tools.dt(f) / 1000.0

                    spectra = []
                    for idx in sample_indices:
                        trace = f.trace[idx]
                        if np.std(trace) > 0:
                            # Compute spectrum
                            n = len(trace)
                            window = np.hanning(n)
                            windowed = trace * window
                            spectrum = np.abs(np.fft.fft(windowed))[:n//2]
                            spectra.append(spectrum)

                    if spectra:
                        avg_spectrum = np.mean(spectra, axis=0)
                        freqs = np.fft.fftfreq(len(trace), dt_ms / 1000)[:len(trace)//2]

                        # Find dominant frequency
                        dom_idx = np.argmax(avg_spectrum)
                        dom_freq = freqs[dom_idx]

                        # Find bandwidth (-6dB points)
                        max_amp = np.max(avg_spectrum)
                        threshold = max_amp * 0.5
                        above_threshold = freqs[avg_spectrum > threshold]

                        if len(above_threshold) > 1:
                            low_freq = np.min(above_threshold)
                            high_freq = np.max(above_threshold)
                            bandwidth = high_freq - low_freq
                        else:
                            low_freq, high_freq, bandwidth = 0, 0, 0

                        line_result = {
                            'line_name': line.line_name,
                            'processing': line.processing_type,
                            'dominant_freq_hz': float(dom_freq),
                            'low_freq_hz': float(low_freq),
                            'high_freq_hz': float(high_freq),
                            'bandwidth_hz': float(bandwidth),
                            'sample_interval_ms': float(dt_ms)
                        }

                        freq_results['per_line'].append(line_result)
                        all_dominant_freqs.append(dom_freq)
                        all_bandwidths.append(bandwidth)

            except Exception as e:
                logger.debug(f"Frequency analysis error for {line.line_name}: {e}")

        # Summary statistics
        if all_dominant_freqs:
            freq_results['summary'] = {
                'mean_dominant_freq_hz': float(np.mean(all_dominant_freqs)),
                'std_dominant_freq_hz': float(np.std(all_dominant_freqs)),
                'mean_bandwidth_hz': float(np.mean(all_bandwidths)),
                'std_bandwidth_hz': float(np.std(all_bandwidths)),
                'frequency_range': [float(np.min(all_dominant_freqs)), float(np.max(all_dominant_freqs))]
            }

        return freq_results

    def _analyze_amplitude_characteristics(self, lines: List[Line2DInfo]) -> Dict[str, Any]:
        """Analyze amplitude characteristics across lines"""
        if not SEGYIO_AVAILABLE:
            return {'error': 'segyio not available'}

        amp_results = {
            'per_line': [],
            'summary': {}
        }

        all_rms = []
        all_max = []
        all_dynamic_range = []

        for line in lines:
            try:
                with segyio.open(line.filepath, "r", ignore_geometry=True) as f:
                    # Sample traces
                    n_traces = f.tracecount
                    n_sample = min(100, n_traces)
                    sample_indices = np.linspace(0, n_traces - 1, n_sample, dtype=int)

                    amplitudes = []
                    for idx in sample_indices:
                        trace = f.trace[idx]
                        amplitudes.extend(trace)

                    amplitudes = np.array(amplitudes)

                    rms = np.sqrt(np.mean(amplitudes**2))
                    max_amp = np.max(np.abs(amplitudes))
                    min_amp = np.min(np.abs(amplitudes[amplitudes != 0])) if np.any(amplitudes != 0) else 1e-10
                    dynamic_range_db = 20 * np.log10(max_amp / (min_amp + 1e-10))

                    line_result = {
                        'line_name': line.line_name,
                        'processing': line.processing_type,
                        'rms_amplitude': float(rms),
                        'max_amplitude': float(max_amp),
                        'dynamic_range_db': float(dynamic_range_db),
                        'percentile_99': float(np.percentile(np.abs(amplitudes), 99))
                    }

                    amp_results['per_line'].append(line_result)
                    all_rms.append(rms)
                    all_max.append(max_amp)
                    all_dynamic_range.append(dynamic_range_db)

            except Exception as e:
                logger.debug(f"Amplitude analysis error for {line.line_name}: {e}")

        if all_rms:
            amp_results['summary'] = {
                'mean_rms': float(np.mean(all_rms)),
                'std_rms': float(np.std(all_rms)),
                'rms_variation_pct': float(np.std(all_rms) / np.mean(all_rms) * 100) if np.mean(all_rms) > 0 else 0,
                'mean_dynamic_range_db': float(np.mean(all_dynamic_range)),
                'amplitude_consistency': 'GOOD' if np.std(all_rms) / np.mean(all_rms) < 0.3 else 'VARIABLE'
            }

        return amp_results

    def _compare_mig_stk(self) -> Dict[str, Any]:
        """Compare MIG and STK versions where both exist"""
        comparison = {
            'pairs_found': [],
            'summary': {}
        }

        # Find lines with both MIG and STK
        mig_lines = {l.survey_id: l for l in self.line_inventory if l.processing_type == 'MIG'}
        stk_lines = {l.survey_id: l for l in self.line_inventory if l.processing_type == 'STK'}

        common_surveys = set(mig_lines.keys()) & set(stk_lines.keys())

        for survey in list(common_surveys)[:5]:  # Limit to 5 comparisons
            mig = mig_lines[survey]
            stk = stk_lines[survey]

            pair_info = {
                'survey_id': survey,
                'mig_line': mig.line_name,
                'stk_line': stk.line_name,
                'mig_traces': mig.n_traces,
                'stk_traces': stk.n_traces,
                'mig_quality': mig.quality_tier,
                'stk_quality': stk.quality_tier,
                'mig_rms': mig.rms_amplitude,
                'stk_rms': stk.rms_amplitude
            }

            comparison['pairs_found'].append(pair_info)

        comparison['summary'] = {
            'n_mig_lines': len(mig_lines),
            'n_stk_lines': len(stk_lines),
            'n_pairs_with_both': len(common_surveys),
            'recommendation': 'Use MIG for interpretation (better imaging)' if len(mig_lines) > 0 else 'Use STK (no MIG available)'
        }

        return comparison

    def _generate_eda_recommendations(self, eda_results: Dict) -> List[str]:
        """Generate recommendations based on EDA findings"""
        recs = []

        # Processing state recommendations
        proc_summary = eda_results.get('processing_state_analysis', {})
        if proc_summary.get('mig_available'):
            recs.append("MIGRATED (MIG) lines available - prefer these for structural interpretation")
        if not proc_summary.get('mig_available') and proc_summary.get('stk_available'):
            recs.append("Only STACKED (STK) lines available - acceptable for regional interpretation")

        # Frequency recommendations
        freq_summary = eda_results.get('frequency_analysis', {}).get('summary', {})
        if freq_summary:
            dom_freq = freq_summary.get('mean_dominant_freq_hz', 0)
            if dom_freq > 0:
                # Estimate vertical resolution (lambda/4)
                avg_velocity = 3000  # Assume ~3000 m/s
                wavelength = avg_velocity / dom_freq
                vertical_res = wavelength / 4
                recs.append(f"Dominant frequency ~{dom_freq:.0f} Hz, estimated vertical resolution ~{vertical_res:.0f} m")

        # Amplitude recommendations
        amp_summary = eda_results.get('amplitude_analysis', {}).get('summary', {})
        if amp_summary:
            if amp_summary.get('amplitude_consistency') == 'VARIABLE':
                recs.append("Amplitude levels vary between lines - apply consistent scaling before integration")
            else:
                recs.append("Amplitude levels are consistent across lines - good for regional comparison")

        # General recommendations
        recs.append("Document processing history in thesis methodology chapter")
        recs.append("Compare 2D frequency content with 3D before integration")

        return recs

    def _create_eda_figures(self, eda_results: Dict, rep_lines: List[Line2DInfo]):
        """Create comprehensive EDA figures"""
        fig_dir = Path(self.config.output_dir) / "eda_figures"
        fig_dir.mkdir(exist_ok=True)

        # Figure 1: Processing State Distribution
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        proc_breakdown = eda_results['processing_state_analysis'].get('processing_breakdown', {})
        if proc_breakdown:
            ax1 = axes[0]
            ax1.bar(proc_breakdown.keys(), proc_breakdown.values(), color=['green', 'blue', 'orange', 'gray'][:len(proc_breakdown)])
            ax1.set_xlabel('Processing Type')
            ax1.set_ylabel('Number of Lines')
            ax1.set_title('2D Line Processing State Distribution')
            for i, (k, v) in enumerate(proc_breakdown.items()):
                ax1.text(i, v + 0.5, str(v), ha='center', fontweight='bold')

        # Processing type recommendation
        ax2 = axes[1]
        ax2.axis('off')
        rec_text = "PROCESSING STATE SUMMARY\n\n"
        for k, v in proc_breakdown.items():
            rec_text += f"{k}: {v} lines\n"
        rec_text += f"\nRecommendation: {eda_results['processing_state_analysis'].get('recommended_for_interpretation', 'N/A')}"
        ax2.text(0.1, 0.5, rec_text, fontsize=12, verticalalignment='center',
                fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightgray'))

        plt.suptitle('2D Seismic EDA: Processing Analysis', fontweight='bold')
        plt.tight_layout()
        fig.savefig(fig_dir / 'eda_processing_state.png', dpi=self.config.figure_dpi, bbox_inches='tight')
        plt.close()

        # Figure 2: Frequency Analysis
        freq_data = eda_results.get('frequency_analysis', {}).get('per_line', [])
        if freq_data:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            ax1 = axes[0]
            line_names = [d['line_name'][:15] for d in freq_data]
            dom_freqs = [d['dominant_freq_hz'] for d in freq_data]
            ax1.barh(line_names, dom_freqs, color='steelblue')
            ax1.set_xlabel('Dominant Frequency (Hz)')
            ax1.set_title('Dominant Frequency by Line')
            ax1.grid(True, alpha=0.3)

            ax2 = axes[1]
            bandwidths = [d['bandwidth_hz'] for d in freq_data]
            ax2.barh(line_names, bandwidths, color='teal')
            ax2.set_xlabel('Bandwidth (Hz)')
            ax2.set_title('Usable Bandwidth by Line')
            ax2.grid(True, alpha=0.3)

            plt.suptitle('2D Seismic EDA: Frequency Analysis', fontweight='bold')
            plt.tight_layout()
            fig.savefig(fig_dir / 'eda_frequency_analysis.png', dpi=self.config.figure_dpi, bbox_inches='tight')
            plt.close()

        # Figure 3: Amplitude Analysis
        amp_data = eda_results.get('amplitude_analysis', {}).get('per_line', [])
        if amp_data:
            fig, ax = plt.subplots(figsize=(10, 6))

            line_names = [d['line_name'][:15] for d in amp_data]
            rms_values = [d['rms_amplitude'] for d in amp_data]

            ax.bar(line_names, rms_values, color='coral')
            ax.set_xlabel('Line')
            ax.set_ylabel('RMS Amplitude')
            ax.set_title('RMS Amplitude by Line')
            ax.tick_params(axis='x', rotation=45)

            # Add mean line
            mean_rms = np.mean(rms_values)
            ax.axhline(mean_rms, color='red', linestyle='--', label=f'Mean: {mean_rms:.2e}')
            ax.legend()

            plt.tight_layout()
            fig.savefig(fig_dir / 'eda_amplitude_analysis.png', dpi=self.config.figure_dpi, bbox_inches='tight')
            plt.close()

        # Figure 4: Representative line displays
        for line in rep_lines[:3]:  # Display top 3
            self.create_line_display(line.line_name)

        self.results.output_files['eda_figures'] = str(fig_dir)
        logger.info(f"EDA figures saved to {fig_dir}")

    # =========================================================================
    # EXPORT
    # =========================================================================

    def export_inventory(self):
        """Export inventory to CSV"""
        logger.info("Exporting inventory...")

        if not PANDAS_AVAILABLE:
            logger.warning("pandas not available for CSV export")
            return

        # Line inventory
        data = []
        for line in self.line_inventory:
            data.append({
                "Line_Name": line.line_name,
                "Survey_ID": line.survey_id,
                "Processing": line.processing_type,
                "N_Traces": line.n_traces,
                "N_Samples": line.n_samples,
                "Sample_Interval_ms": line.sample_interval_ms,
                "Record_Length_ms": line.record_length_ms,
                "Start_X": line.start_x,
                "Start_Y": line.start_y,
                "End_X": line.end_x,
                "End_Y": line.end_y,
                "Line_Length_m": line.line_length_m,
                "CDP_Start": line.cdp_start,
                "CDP_End": line.cdp_end,
                "Dead_Trace_Pct": line.dead_trace_pct,
                "RMS_Amplitude": line.rms_amplitude,
                "Quality_Tier": line.quality_tier,
                "Status": line.status,
                "Filename": line.filename
            })

        df = pd.DataFrame(data)
        csv_path = Path(self.config.output_dir) / "line_inventory.csv"
        df.to_csv(csv_path, index=False)

        self.results.output_files['inventory_csv'] = str(csv_path)
        logger.info(f"Inventory exported to {csv_path}")

        # Well ties
        if self.results.well_ties:
            tie_data = []
            for tie in self.results.well_ties:
                for nearest in tie["nearest_lines"]:
                    tie_data.append({
                        "Well_Name": tie["well_name"],
                        "Well_X": tie["well_x"],
                        "Well_Y": tie["well_y"],
                        "Line_Name": nearest["line_name"],
                        "Distance_m": nearest["distance_m"],
                        "Processing": nearest["processing_type"],
                        "Quality": nearest["quality_tier"]
                    })

            if tie_data:
                df_ties = pd.DataFrame(tie_data)
                ties_path = Path(self.config.output_dir) / "well_line_ties.csv"
                df_ties.to_csv(ties_path, index=False)
                self.results.output_files['well_ties_csv'] = str(ties_path)
                logger.info(f"Well ties exported to {ties_path}")

    # =========================================================================
    # MAIN PIPELINE
    # =========================================================================

    def run(self, quick_scan: bool = False, run_eda: bool = True) -> Seismic2DResults:
        """Execute the complete 2D seismic processing pipeline"""
        start_time = datetime.now()

        logger.info("=" * 80)
        logger.info("2D SEISMIC PROCESSING AND INTEGRATION MODULE v1.0")
        logger.info("PhD-Level Analysis for Bornu Chad Basin")
        logger.info("=" * 80)

        self.results.timestamp = start_time.isoformat()
        self.results.input_directory = self.config.input_directory

        # Step 1: Build inventory
        logger.info("\n[Step 1/6] Building line inventory...")
        self.build_inventory(quick_scan=quick_scan)

        # Step 2: Comprehensive EDA (PhD-level analysis)
        if run_eda and not quick_scan:
            logger.info("\n[Step 2/6] Running comprehensive EDA...")
            self.run_comprehensive_eda()
        else:
            logger.info("\n[Step 2/6] Skipping EDA (quick scan mode)")

        # Step 3: Load well locations
        logger.info("\n[Step 3/6] Loading well locations...")
        self.load_well_locations()

        # Step 4: Analyze well ties
        logger.info("\n[Step 4/6] Analyzing well-to-2D ties...")
        self.analyze_well_ties()

        # Step 5: Create visualizations
        if self.config.save_figures:
            logger.info("\n[Step 5/6] Creating visualizations...")
            self.create_inventory_plot()
            if self.config.create_basemap:
                self.create_basemap()

        # Step 6: Export data
        if self.config.export_inventory_csv:
            logger.info("\n[Step 6/6] Exporting data...")
            self.export_inventory()

        # Finalize
        end_time = datetime.now()
        self.results.processing_time_seconds = (end_time - start_time).total_seconds()

        # Generate recommendations
        self._generate_recommendations()

        # Save results
        results_file = Path(self.config.output_dir) / "2d_seismic_results.json"
        self.results.to_json(str(results_file))
        self.results.output_files['results_json'] = str(results_file)

        # Save config
        config_file = Path(self.config.output_dir) / "2d_config_used.json"
        self.config.to_json(str(config_file))

        logger.info("\n" + "=" * 80)
        logger.info("2D SEISMIC PROCESSING COMPLETE")
        logger.info(f"  Lines processed: {self.results.lines_processed}")
        logger.info(f"  Lines failed: {self.results.lines_failed}")
        logger.info(f"  Results: {results_file}")
        logger.info(f"  Time: {self.results.processing_time_seconds:.1f}s")
        logger.info("=" * 80)

        return self.results

    def _generate_recommendations(self):
        """Generate recommendations based on analysis"""
        recs = []
        warns = []

        # Quality recommendations
        good_lines = sum(1 for l in self.line_inventory if l.quality_tier == "GOOD")
        total_lines = len(self.line_inventory)

        if good_lines < total_lines * 0.5:
            warns.append(f"Less than 50% of lines are good quality ({good_lines}/{total_lines})")
            recs.append("Consider additional QC on fair/poor quality lines")

        # Well tie recommendations
        if self.results.well_ties:
            wells_with_ties = len([t for t in self.results.well_ties if t["n_lines_within_5km"] > 0])
            total_wells = len(self.well_locations)

            if wells_with_ties < total_wells:
                warns.append(f"Only {wells_with_ties}/{total_wells} wells have 2D lines within 5km")
                recs.append("Use 3D seismic for wells without nearby 2D coverage")

        # Processing type recommendations
        survey_summary = self.results.survey_summary
        for survey_id, info in survey_summary.items():
            if "MIG" in info.get("processing_types", []) and "STK" in info.get("processing_types", []):
                recs.append(f"Survey {survey_id}: Both MIG and STK available - prefer MIG for interpretation")

        # Integration recommendations
        recs.append("Create composite 2D/3D horizon maps for regional framework")
        recs.append("Use 2D lines for regional velocity trend validation")
        recs.append("Consider 2D lines for fault framework interpretation beyond 3D coverage")

        self.results.recommendations = recs
        self.results.warnings = warns


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description="2D Seismic Processing and Integration Module v1.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with directory
    python seismic_2d_automation.py "path/to/2d_folder" -o "outputs"

    # Quick inventory scan
    python seismic_2d_automation.py "path/to/2d_folder" --inventory

    # Run with config file
    python seismic_2d_automation.py -c config.json

    # Generate default config
    python seismic_2d_automation.py --create-config my_config.json
        """
    )

    parser.add_argument("input_directory", nargs="?", help="Directory containing 2D SEGY files")
    parser.add_argument("-o", "--output-dir", default="2d_outputs", help="Output directory")
    parser.add_argument("-c", "--config", help="Configuration JSON file")
    parser.add_argument("--inventory", action="store_true", help="Quick inventory scan only")
    parser.add_argument("--well-header", help="Well header Excel/CSV file")
    parser.add_argument("--las-dir", help="LAS files directory")
    parser.add_argument("--no-figures", action="store_true", help="Skip figure generation")
    parser.add_argument("--create-config", metavar="FILE", help="Generate default config file")
    parser.add_argument("--display-line", help="Create display for specific line")

    args = parser.parse_args()

    # Generate config file
    if args.create_config:
        config = Seismic2DConfig()
        config.to_json(args.create_config)
        print(f"Created config: {args.create_config}")
        return

    # Load or create config
    if args.config:
        config = Seismic2DConfig.from_json(args.config)
    else:
        if not args.input_directory:
            parser.error("Either input_directory or --config is required")
        config = Seismic2DConfig(input_directory=args.input_directory)

    # Override with CLI arguments
    if args.input_directory:
        config.input_directory = args.input_directory
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.well_header:
        config.well_header_file = args.well_header
    if args.las_dir:
        config.las_directory = args.las_dir
    if args.no_figures:
        config.save_figures = False

    # Run
    automation = Seismic2DAutomation(config)

    if args.display_line:
        automation.build_inventory(quick_scan=True)
        automation.create_line_display(args.display_line)
    else:
        results = automation.run(quick_scan=args.inventory)

        # Print summary
        print(f"\nLines processed: {results.lines_processed}")
        print(f"Usable lines: {results.quality_summary.get('usable_lines', 0)}")

        if results.recommendations:
            print("\nRecommendations:")
            for rec in results.recommendations:
                print(f"  - {rec}")

    return results


if __name__ == "__main__":
    main()
