"""
================================================================================
2D-3D SEISMIC INTEGRATION FRAMEWORK v1.0
PhD Research - Comparative Analysis and Regional Integration
================================================================================

Author: Moses Ekene Obasi
Institution: University of Calabar, Nigeria
Supervisor: Prof. Dominic Akam Obi

Purpose:
This module provides comprehensive 2D-3D seismic integration for:
- Comparative amplitude and frequency analysis
- Regional structural framework extension
- Velocity model validation across datasets
- Composite horizon mapping
- Data quality cross-validation

Key Features:
1. Geometric Integration - Find 2D lines intersecting 3D survey
2. Amplitude Comparison - Compare amplitude response at tie points
3. Frequency Analysis - Spectral comparison between datasets
4. Velocity Validation - Cross-validate time-depth relationships
5. Horizon Correlation - Extend 3D horizons into 2D regional coverage
6. Composite Mapping - Generate unified structure maps

Usage:
    python integration_2d3d_automation.py -c config.json
    python integration_2d3d_automation.py --survey-overlap  # Find overlaps
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from matplotlib.collections import PatchCollection
from scipy import signal, interpolate
from scipy.fft import fft, fftfreq
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import logging
import json
import warnings

warnings.filterwarnings('ignore')

# Try imports
try:
    import segyio
    SEGYIO_AVAILABLE = True
except ImportError:
    SEGYIO_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Integration2D3DConfig:
    """Configuration for 2D-3D integration"""

    # Input paths
    segy_3d_path: str = ""
    segy_2d_directory: str = ""
    line_inventory_csv: str = ""  # From 2D module output

    # Well data
    well_header_file: str = ""
    las_directory: str = ""
    velocity_model_path: str = ""  # From well integration output

    # Output
    output_dir: str = "integration_outputs"

    # 3D survey geometry (will be auto-detected if not specified)
    survey_3d_bounds: Dict[str, float] = field(default_factory=lambda: {
        "xmin": 0, "xmax": 0, "ymin": 0, "ymax": 0
    })

    # Analysis parameters
    comparison_window_ms: float = 100.0  # Window for amplitude comparison
    frequency_analysis_window_ms: float = 500.0
    max_tie_distance_m: float = 100.0  # Max distance for amplitude comparison

    # Velocity analysis
    velocity_comparison_depths: List[float] = field(default_factory=lambda: [
        1000, 1500, 2000, 2500, 3000, 3500, 4000
    ])

    # Horizon correlation
    correlation_horizons: List[str] = field(default_factory=lambda: [
        "Chad_Top", "Fika_Top", "Gongila_Top", "Bima_Top"
    ])

    # Visualization
    save_figures: bool = True
    figure_dpi: int = 300

    @classmethod
    def from_json(cls, path: str) -> 'Integration2D3DConfig':
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)

    def to_json(self, path: str):
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)


@dataclass
class IntegrationResults:
    """Results from 2D-3D integration analysis"""

    timestamp: str = ""
    version: str = "1.0"
    processing_time_seconds: float = 0.0

    # Survey geometry
    survey_3d_bounds: Dict[str, float] = field(default_factory=dict)
    survey_3d_area_km2: float = 0.0

    # Line intersections
    intersecting_lines: List[Dict[str, Any]] = field(default_factory=list)
    n_intersecting_lines: int = 0

    # Amplitude comparison
    amplitude_comparison: Dict[str, Any] = field(default_factory=dict)

    # Frequency analysis
    frequency_analysis: Dict[str, Any] = field(default_factory=dict)

    # Velocity validation
    velocity_validation: Dict[str, Any] = field(default_factory=dict)

    # Quality metrics
    integration_quality: Dict[str, Any] = field(default_factory=dict)

    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Output files
    output_files: Dict[str, str] = field(default_factory=dict)

    def to_json(self, path: str):
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2, default=str)


# =============================================================================
# GEOMETRY UTILITIES
# =============================================================================

class GeometryUtils:
    """Utilities for geometric calculations"""

    @staticmethod
    def line_intersects_box(x1: float, y1: float, x2: float, y2: float,
                            box: Dict[str, float]) -> Tuple[bool, Optional[List[Tuple[float, float]]]]:
        """
        Check if a line segment intersects a bounding box.
        Returns (intersects, intersection_points)
        """
        xmin, xmax = box['xmin'], box['xmax']
        ymin, ymax = box['ymin'], box['ymax']

        # Cohen-Sutherland algorithm
        def compute_code(x, y):
            code = 0
            if x < xmin: code |= 1
            elif x > xmax: code |= 2
            if y < ymin: code |= 4
            elif y > ymax: code |= 8
            return code

        code1 = compute_code(x1, y1)
        code2 = compute_code(x2, y2)

        intersections = []

        while True:
            if code1 == 0 and code2 == 0:
                # Both inside
                return True, [(x1, y1), (x2, y2)]
            elif code1 & code2 != 0:
                # Both outside on same side
                return False, None
            else:
                # Some intersection
                code_out = code1 if code1 != 0 else code2

                if code_out & 8:
                    x = x1 + (x2 - x1) * (ymax - y1) / (y2 - y1 + 1e-10)
                    y = ymax
                elif code_out & 4:
                    x = x1 + (x2 - x1) * (ymin - y1) / (y2 - y1 + 1e-10)
                    y = ymin
                elif code_out & 2:
                    y = y1 + (y2 - y1) * (xmax - x1) / (x2 - x1 + 1e-10)
                    x = xmax
                elif code_out & 1:
                    y = y1 + (y2 - y1) * (xmin - x1) / (x2 - x1 + 1e-10)
                    x = xmin

                if code_out == code1:
                    x1, y1 = x, y
                    code1 = compute_code(x1, y1)
                    intersections.append((x, y))
                else:
                    x2, y2 = x, y
                    code2 = compute_code(x2, y2)
                    intersections.append((x, y))

                if len(intersections) >= 2:
                    return True, intersections

    @staticmethod
    def point_to_line_projection(px: float, py: float,
                                  x1: float, y1: float,
                                  x2: float, y2: float) -> Tuple[float, float, float]:
        """
        Project point onto line segment.
        Returns (proj_x, proj_y, distance)
        """
        dx = x2 - x1
        dy = y2 - y1
        l2 = dx*dx + dy*dy

        if l2 == 0:
            return x1, y1, np.sqrt((px-x1)**2 + (py-y1)**2)

        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / l2))
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        dist = np.sqrt((px - proj_x)**2 + (py - proj_y)**2)

        return proj_x, proj_y, dist


# =============================================================================
# MAIN INTEGRATION CLASS
# =============================================================================

class Integration2D3D:
    """
    2D-3D Seismic Integration Framework

    Workflow:
    1. Extract 3D survey geometry
    2. Find 2D lines intersecting/near 3D survey
    3. Extract traces at tie points
    4. Compare amplitude characteristics
    5. Compare frequency content
    6. Validate velocity models
    7. Generate composite interpretations
    """

    def __init__(self, config: Integration2D3DConfig):
        self.config = config
        self.results = IntegrationResults()
        self.geometry = GeometryUtils()

        # Data storage
        self.survey_bounds = {}
        self.line_inventory = []
        self.intersecting_lines = []

        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # STEP 1: EXTRACT 3D SURVEY GEOMETRY
    # =========================================================================

    def extract_3d_geometry(self) -> Dict[str, float]:
        """Extract 3D survey bounding box from SEGY headers"""
        logger.info("Extracting 3D survey geometry...")

        if not SEGYIO_AVAILABLE:
            logger.error("segyio not available")
            return {}

        if not self.config.segy_3d_path:
            logger.warning("No 3D SEGY path specified")
            return {}

        try:
            with segyio.open(self.config.segy_3d_path, "r", ignore_geometry=True) as f:
                n_traces = f.tracecount

                # Sample headers to find bounds (full scan is slow for large files)
                sample_size = min(10000, n_traces)
                indices = np.linspace(0, n_traces - 1, sample_size, dtype=int)

                xs, ys = [], []

                for idx in tqdm(indices, desc="Scanning 3D headers"):
                    header = f.header[idx]

                    scalar = header.get(segyio.TraceField.SourceGroupScalar, 1)
                    if scalar < 0:
                        scalar = -1.0 / scalar
                    elif scalar == 0:
                        scalar = 1

                    x = header.get(segyio.TraceField.CDP_X, 0) * scalar
                    y = header.get(segyio.TraceField.CDP_Y, 0) * scalar

                    if x != 0 and y != 0:
                        xs.append(x)
                        ys.append(y)

                if xs and ys:
                    bounds = {
                        'xmin': min(xs),
                        'xmax': max(xs),
                        'ymin': min(ys),
                        'ymax': max(ys)
                    }

                    # Calculate area
                    area_m2 = (bounds['xmax'] - bounds['xmin']) * (bounds['ymax'] - bounds['ymin'])
                    area_km2 = area_m2 / 1e6

                    self.survey_bounds = bounds
                    self.results.survey_3d_bounds = bounds
                    self.results.survey_3d_area_km2 = area_km2

                    logger.info(f"3D Survey bounds: X=[{bounds['xmin']:.0f}, {bounds['xmax']:.0f}], "
                              f"Y=[{bounds['ymin']:.0f}, {bounds['ymax']:.0f}]")
                    logger.info(f"Survey area: {area_km2:.1f} km2")

                    return bounds

        except Exception as e:
            logger.error(f"Error extracting 3D geometry: {e}")

        return {}

    # =========================================================================
    # STEP 2: FIND INTERSECTING 2D LINES
    # =========================================================================

    def load_2d_inventory(self) -> List[Dict]:
        """Load 2D line inventory from CSV"""
        if not self.config.line_inventory_csv:
            logger.warning("No 2D inventory CSV specified")
            return []

        if not PANDAS_AVAILABLE:
            logger.warning("pandas not available")
            return []

        try:
            df = pd.read_csv(self.config.line_inventory_csv)
            self.line_inventory = df.to_dict('records')
            logger.info(f"Loaded {len(self.line_inventory)} lines from inventory")
            return self.line_inventory
        except Exception as e:
            logger.error(f"Error loading inventory: {e}")
            return []

    def find_intersecting_lines(self) -> List[Dict]:
        """Find 2D lines that intersect the 3D survey area"""
        logger.info("Finding 2D lines intersecting 3D survey...")

        if not self.survey_bounds:
            self.extract_3d_geometry()

        if not self.survey_bounds:
            logger.error("No 3D bounds available")
            return []

        if not self.line_inventory:
            self.load_2d_inventory()

        intersecting = []

        for line in self.line_inventory:
            # Check if line has coordinates
            start_x = line.get('Start_X')
            start_y = line.get('Start_Y')
            end_x = line.get('End_X')
            end_y = line.get('End_Y')

            if None in [start_x, start_y, end_x, end_y]:
                continue

            if any(pd.isna([start_x, start_y, end_x, end_y])):
                continue

            # Check intersection
            intersects, points = self.geometry.line_intersects_box(
                start_x, start_y, end_x, end_y, self.survey_bounds
            )

            if intersects and points:
                # Calculate intersection length
                if len(points) >= 2:
                    int_length = np.sqrt(
                        (points[1][0] - points[0][0])**2 +
                        (points[1][1] - points[0][1])**2
                    )
                else:
                    int_length = 0

                intersecting.append({
                    'line_name': line.get('Line_Name', ''),
                    'filename': line.get('Filename', ''),
                    'processing': line.get('Processing', ''),
                    'quality': line.get('Quality_Tier', ''),
                    'intersection_points': points,
                    'intersection_length_m': int_length,
                    'full_line_length_m': line.get('Line_Length_m', 0),
                    'n_traces': line.get('N_Traces', 0)
                })

        # Sort by intersection length
        intersecting.sort(key=lambda x: x['intersection_length_m'], reverse=True)

        self.intersecting_lines = intersecting
        self.results.intersecting_lines = intersecting
        self.results.n_intersecting_lines = len(intersecting)

        logger.info(f"Found {len(intersecting)} lines intersecting 3D survey")

        return intersecting

    # =========================================================================
    # STEP 3: AMPLITUDE COMPARISON
    # =========================================================================

    def compare_amplitudes(self) -> Dict[str, Any]:
        """Compare amplitude characteristics between 2D and 3D at tie points"""
        logger.info("Comparing amplitude characteristics...")

        if not self.intersecting_lines:
            logger.warning("No intersecting lines found")
            return {}

        if not SEGYIO_AVAILABLE:
            return {}

        comparison_results = {
            'tie_points': [],
            'amplitude_ratios': [],
            'correlation_coefficients': [],
            'summary': {}
        }

        # For each intersecting line, extract and compare traces
        for line_info in tqdm(self.intersecting_lines[:10], desc="Comparing amplitudes"):
            try:
                # Get intersection midpoint
                points = line_info.get('intersection_points', [])
                if len(points) < 2:
                    continue

                mid_x = (points[0][0] + points[1][0]) / 2
                mid_y = (points[0][1] + points[1][1]) / 2

                # Extract 2D trace at midpoint
                trace_2d = self._extract_2d_trace_at_location(
                    line_info['filename'], mid_x, mid_y
                )

                # Extract 3D trace at same location
                trace_3d = self._extract_3d_trace_at_location(mid_x, mid_y)

                if trace_2d is not None and trace_3d is not None:
                    # Resample to common length if needed
                    min_len = min(len(trace_2d), len(trace_3d))
                    trace_2d = trace_2d[:min_len]
                    trace_3d = trace_3d[:min_len]

                    # Calculate metrics
                    rms_2d = np.sqrt(np.mean(trace_2d**2))
                    rms_3d = np.sqrt(np.mean(trace_3d**2))
                    ratio = rms_2d / (rms_3d + 1e-10)

                    # Correlation
                    if np.std(trace_2d) > 0 and np.std(trace_3d) > 0:
                        corr = np.corrcoef(trace_2d, trace_3d)[0, 1]
                    else:
                        corr = 0

                    comparison_results['tie_points'].append({
                        'line_name': line_info['line_name'],
                        'x': mid_x,
                        'y': mid_y,
                        'rms_2d': float(rms_2d),
                        'rms_3d': float(rms_3d),
                        'amplitude_ratio': float(ratio),
                        'correlation': float(corr)
                    })

                    comparison_results['amplitude_ratios'].append(ratio)
                    comparison_results['correlation_coefficients'].append(corr)

            except Exception as e:
                logger.debug(f"Error comparing {line_info['line_name']}: {e}")

        # Summary statistics
        if comparison_results['amplitude_ratios']:
            ratios = np.array(comparison_results['amplitude_ratios'])
            corrs = np.array(comparison_results['correlation_coefficients'])

            comparison_results['summary'] = {
                'n_tie_points': len(ratios),
                'mean_amplitude_ratio': float(np.mean(ratios)),
                'std_amplitude_ratio': float(np.std(ratios)),
                'mean_correlation': float(np.mean(corrs)),
                'std_correlation': float(np.std(corrs)),
                'scaling_factor_2d_to_3d': float(1.0 / np.mean(ratios)) if np.mean(ratios) > 0 else 1.0
            }

            logger.info(f"Amplitude comparison: ratio={np.mean(ratios):.2f} +/- {np.std(ratios):.2f}, "
                       f"correlation={np.mean(corrs):.2f}")

        self.results.amplitude_comparison = comparison_results
        return comparison_results

    def _extract_2d_trace_at_location(self, filename: str, x: float, y: float) -> Optional[np.ndarray]:
        """Extract trace from 2D line nearest to given location"""
        # Build full path
        filepath = Path(self.config.segy_2d_directory) / filename

        if not filepath.exists():
            # Try finding it
            for ext in ['', '.sgy', '.SGY', '.segy', '.SEGY']:
                test_path = Path(self.config.segy_2d_directory) / (filename + ext)
                if test_path.exists():
                    filepath = test_path
                    break

        if not filepath.exists():
            return None

        try:
            with segyio.open(str(filepath), "r", ignore_geometry=True) as f:
                # Find nearest trace
                min_dist = float('inf')
                nearest_idx = 0

                # Sample traces for speed
                n_traces = f.tracecount
                sample_indices = np.linspace(0, n_traces - 1, min(100, n_traces), dtype=int)

                for idx in sample_indices:
                    header = f.header[idx]
                    scalar = header.get(segyio.TraceField.SourceGroupScalar, 1)
                    if scalar < 0:
                        scalar = -1.0 / scalar
                    elif scalar == 0:
                        scalar = 1

                    tx = header.get(segyio.TraceField.CDP_X, 0) * scalar
                    ty = header.get(segyio.TraceField.CDP_Y, 0) * scalar

                    dist = np.sqrt((tx - x)**2 + (ty - y)**2)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_idx = idx

                if min_dist <= self.config.max_tie_distance_m * 10:  # Allow some tolerance
                    return np.array(f.trace[nearest_idx])

        except Exception as e:
            logger.debug(f"Error extracting 2D trace: {e}")

        return None

    def _extract_3d_trace_at_location(self, x: float, y: float) -> Optional[np.ndarray]:
        """Extract trace from 3D volume nearest to given location"""
        if not self.config.segy_3d_path:
            return None

        try:
            with segyio.open(self.config.segy_3d_path, "r", ignore_geometry=True) as f:
                # Sample traces to find nearest
                n_traces = f.tracecount
                sample_size = min(5000, n_traces)
                sample_indices = np.linspace(0, n_traces - 1, sample_size, dtype=int)

                min_dist = float('inf')
                nearest_idx = 0

                for idx in sample_indices:
                    header = f.header[idx]
                    scalar = header.get(segyio.TraceField.SourceGroupScalar, 1)
                    if scalar < 0:
                        scalar = -1.0 / scalar
                    elif scalar == 0:
                        scalar = 1

                    tx = header.get(segyio.TraceField.CDP_X, 0) * scalar
                    ty = header.get(segyio.TraceField.CDP_Y, 0) * scalar

                    dist = np.sqrt((tx - x)**2 + (ty - y)**2)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_idx = idx

                if min_dist <= self.config.max_tie_distance_m * 10:
                    return np.array(f.trace[nearest_idx])

        except Exception as e:
            logger.debug(f"Error extracting 3D trace: {e}")

        return None

    # =========================================================================
    # STEP 4: FREQUENCY ANALYSIS
    # =========================================================================

    def compare_frequency_content(self) -> Dict[str, Any]:
        """Compare frequency content between 2D and 3D data"""
        logger.info("Comparing frequency content...")

        freq_results = {
            '2d_spectra': [],
            '3d_spectra': [],
            'comparison': []
        }

        # This would extract and compare spectra at tie points
        # Simplified version for now
        for line_info in self.intersecting_lines[:5]:
            try:
                points = line_info.get('intersection_points', [])
                if len(points) < 2:
                    continue

                mid_x = (points[0][0] + points[1][0]) / 2
                mid_y = (points[0][1] + points[1][1]) / 2

                trace_2d = self._extract_2d_trace_at_location(line_info['filename'], mid_x, mid_y)
                trace_3d = self._extract_3d_trace_at_location(mid_x, mid_y)

                if trace_2d is not None and trace_3d is not None:
                    # Compute spectra
                    spec_2d = self._compute_amplitude_spectrum(trace_2d)
                    spec_3d = self._compute_amplitude_spectrum(trace_3d)

                    freq_results['comparison'].append({
                        'line_name': line_info['line_name'],
                        'dominant_freq_2d_hz': spec_2d.get('dominant_freq', 0),
                        'dominant_freq_3d_hz': spec_3d.get('dominant_freq', 0),
                        'bandwidth_2d_hz': spec_2d.get('bandwidth', 0),
                        'bandwidth_3d_hz': spec_3d.get('bandwidth', 0)
                    })

            except Exception as e:
                logger.debug(f"Frequency comparison error: {e}")

        if freq_results['comparison']:
            # Summary
            dom_2d = [c['dominant_freq_2d_hz'] for c in freq_results['comparison']]
            dom_3d = [c['dominant_freq_3d_hz'] for c in freq_results['comparison']]

            freq_results['summary'] = {
                'mean_dominant_freq_2d': float(np.mean(dom_2d)) if dom_2d else 0,
                'mean_dominant_freq_3d': float(np.mean(dom_3d)) if dom_3d else 0,
                'frequency_ratio': float(np.mean(dom_2d) / (np.mean(dom_3d) + 1e-10)) if dom_3d else 0
            }

        self.results.frequency_analysis = freq_results
        return freq_results

    def _compute_amplitude_spectrum(self, trace: np.ndarray, dt_ms: float = 4.0) -> Dict[str, float]:
        """Compute amplitude spectrum of a trace"""
        n = len(trace)
        if n < 10:
            return {'dominant_freq': 0, 'bandwidth': 0}

        # Apply window
        window = np.hanning(n)
        windowed = trace * window

        # FFT
        spectrum = np.abs(fft(windowed))[:n//2]
        freqs = fftfreq(n, dt_ms/1000)[:n//2]

        # Find dominant frequency
        if len(spectrum) > 0:
            dom_idx = np.argmax(spectrum)
            dom_freq = freqs[dom_idx]

            # Estimate bandwidth (where amplitude > 50% of max)
            threshold = np.max(spectrum) * 0.5
            above_threshold = freqs[spectrum > threshold]
            bandwidth = np.max(above_threshold) - np.min(above_threshold) if len(above_threshold) > 1 else 0

            return {
                'dominant_freq': float(dom_freq),
                'bandwidth': float(bandwidth)
            }

        return {'dominant_freq': 0, 'bandwidth': 0}

    # =========================================================================
    # STEP 5: VISUALIZATION
    # =========================================================================

    def create_integration_map(self):
        """Create map showing 2D-3D integration overview"""
        logger.info("Creating integration map...")

        fig, ax = plt.subplots(figsize=(14, 12))

        # Draw 3D survey box
        if self.survey_bounds:
            bounds = self.survey_bounds
            width = bounds['xmax'] - bounds['xmin']
            height = bounds['ymax'] - bounds['ymin']

            rect = Rectangle(
                (bounds['xmin'], bounds['ymin']),
                width, height,
                linewidth=3, edgecolor='blue', facecolor='lightblue',
                alpha=0.3, label='3D Survey'
            )
            ax.add_patch(rect)

        # Draw all 2D lines
        for line in self.line_inventory:
            start_x = line.get('Start_X')
            start_y = line.get('Start_Y')
            end_x = line.get('End_X')
            end_y = line.get('End_Y')

            if None in [start_x, start_y, end_x, end_y]:
                continue

            ax.plot([start_x, end_x], [start_y, end_y],
                   color='gray', linewidth=0.5, alpha=0.5)

        # Highlight intersecting lines
        for line_info in self.intersecting_lines:
            # Find original line data
            orig = next((l for l in self.line_inventory
                        if l.get('Line_Name') == line_info['line_name']), None)

            if orig:
                ax.plot(
                    [orig['Start_X'], orig['End_X']],
                    [orig['Start_Y'], orig['End_Y']],
                    color='red', linewidth=2, alpha=0.8
                )

                # Mark intersection points
                points = line_info.get('intersection_points', [])
                for pt in points:
                    ax.scatter(pt[0], pt[1], color='yellow', s=50,
                              edgecolors='black', zorder=10)

        # Legend
        ax.plot([], [], color='gray', linewidth=1, label='2D Lines (all)')
        ax.plot([], [], color='red', linewidth=2, label=f'Intersecting Lines ({len(self.intersecting_lines)})')
        ax.scatter([], [], color='yellow', s=50, edgecolors='black', label='Tie Points')

        ax.set_xlabel('Easting (m)', fontweight='bold')
        ax.set_ylabel('Northing (m)', fontweight='bold')
        ax.set_title('2D-3D Seismic Integration Map\nBornu Chad Basin', fontweight='bold', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        plt.tight_layout()

        fig_path = Path(self.config.output_dir) / "integration_map.png"
        fig.savefig(fig_path, dpi=self.config.figure_dpi, bbox_inches='tight')
        plt.close()

        self.results.output_files['integration_map'] = str(fig_path)
        logger.info(f"Integration map saved to {fig_path}")

    def create_amplitude_comparison_plot(self):
        """Create amplitude comparison visualization"""
        if not self.results.amplitude_comparison.get('tie_points'):
            return

        logger.info("Creating amplitude comparison plot...")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        tie_points = self.results.amplitude_comparison['tie_points']

        # 1. Amplitude ratio histogram
        ax1 = axes[0, 0]
        ratios = [tp['amplitude_ratio'] for tp in tie_points]
        ax1.hist(ratios, bins=15, color='steelblue', edgecolor='black', alpha=0.7)
        ax1.axvline(np.mean(ratios), color='red', linestyle='--',
                   label=f'Mean: {np.mean(ratios):.2f}')
        ax1.set_xlabel('Amplitude Ratio (2D/3D)')
        ax1.set_ylabel('Count')
        ax1.set_title('Amplitude Ratio Distribution')
        ax1.legend()

        # 2. Correlation histogram
        ax2 = axes[0, 1]
        corrs = [tp['correlation'] for tp in tie_points]
        ax2.hist(corrs, bins=15, color='green', edgecolor='black', alpha=0.7)
        ax2.axvline(np.mean(corrs), color='red', linestyle='--',
                   label=f'Mean: {np.mean(corrs):.2f}')
        ax2.set_xlabel('Correlation Coefficient')
        ax2.set_ylabel('Count')
        ax2.set_title('2D-3D Trace Correlation')
        ax2.legend()

        # 3. RMS amplitude comparison
        ax3 = axes[1, 0]
        rms_2d = [tp['rms_2d'] for tp in tie_points]
        rms_3d = [tp['rms_3d'] for tp in tie_points]
        ax3.scatter(rms_3d, rms_2d, c='blue', alpha=0.6, s=50)

        # Add 1:1 line
        max_val = max(max(rms_2d), max(rms_3d))
        ax3.plot([0, max_val], [0, max_val], 'k--', label='1:1 line')

        ax3.set_xlabel('RMS Amplitude (3D)')
        ax3.set_ylabel('RMS Amplitude (2D)')
        ax3.set_title('Amplitude Crossplot')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Spatial variation of amplitude ratio
        ax4 = axes[1, 1]
        x_coords = [tp['x'] for tp in tie_points]
        y_coords = [tp['y'] for tp in tie_points]
        ratios = [tp['amplitude_ratio'] for tp in tie_points]

        scatter = ax4.scatter(x_coords, y_coords, c=ratios, cmap='RdYlBu',
                             s=100, edgecolors='black')
        plt.colorbar(scatter, ax=ax4, label='Amplitude Ratio')
        ax4.set_xlabel('Easting (m)')
        ax4.set_ylabel('Northing (m)')
        ax4.set_title('Spatial Amplitude Ratio Variation')

        plt.suptitle('2D-3D Amplitude Comparison Analysis', fontweight='bold', fontsize=14)
        plt.tight_layout()

        fig_path = Path(self.config.output_dir) / "amplitude_comparison.png"
        fig.savefig(fig_path, dpi=self.config.figure_dpi, bbox_inches='tight')
        plt.close()

        self.results.output_files['amplitude_comparison'] = str(fig_path)

    # =========================================================================
    # STEP 6: GENERATE RECOMMENDATIONS
    # =========================================================================

    def generate_recommendations(self):
        """Generate integration recommendations based on analysis"""
        recs = []
        warns = []

        # Based on number of intersecting lines
        n_int = self.results.n_intersecting_lines
        if n_int == 0:
            warns.append("No 2D lines intersect the 3D survey - cannot perform direct comparison")
            recs.append("Use wells for velocity model validation instead of 2D-3D ties")
        elif n_int < 5:
            warns.append(f"Only {n_int} lines intersect 3D - limited tie points available")
        else:
            recs.append(f"Good coverage: {n_int} 2D lines available for tie analysis")

        # Based on amplitude comparison
        amp_summary = self.results.amplitude_comparison.get('summary', {})
        if amp_summary:
            ratio = amp_summary.get('mean_amplitude_ratio', 1.0)
            if ratio < 0.5 or ratio > 2.0:
                warns.append(f"Large amplitude difference between 2D and 3D (ratio={ratio:.2f})")
                recs.append("Apply amplitude scaling before comparative interpretation")
            else:
                recs.append(f"Amplitude levels are compatible (ratio={ratio:.2f})")

            corr = amp_summary.get('mean_correlation', 0)
            if corr < 0.5:
                warns.append(f"Low correlation between 2D and 3D traces (r={corr:.2f})")
                recs.append("Check phase and timing alignment between datasets")
            else:
                recs.append(f"Good waveform correlation (r={corr:.2f}) - data are compatible")

        # General recommendations
        recs.append("Use 2D lines for regional structural framework beyond 3D coverage")
        recs.append("Calibrate 2D interpretation using wells with nearby 2D and 3D coverage")
        recs.append("Create composite horizon maps extending 3D picks into 2D")
        recs.append("Document phase and amplitude differences for thesis methodology chapter")

        self.results.recommendations = recs
        self.results.warnings = warns

    # =========================================================================
    # MAIN PIPELINE
    # =========================================================================

    def run(self) -> IntegrationResults:
        """Execute the complete 2D-3D integration pipeline"""
        start_time = datetime.now()

        logger.info("=" * 80)
        logger.info("2D-3D SEISMIC INTEGRATION FRAMEWORK v1.0")
        logger.info("=" * 80)

        self.results.timestamp = start_time.isoformat()

        # Step 1: Extract 3D geometry
        logger.info("\n[Step 1/6] Extracting 3D survey geometry...")
        self.extract_3d_geometry()

        # Step 2: Load 2D inventory and find intersections
        logger.info("\n[Step 2/6] Finding intersecting 2D lines...")
        self.load_2d_inventory()
        self.find_intersecting_lines()

        # Step 3: Amplitude comparison
        logger.info("\n[Step 3/6] Comparing amplitudes...")
        self.compare_amplitudes()

        # Step 4: Frequency comparison
        logger.info("\n[Step 4/6] Comparing frequency content...")
        self.compare_frequency_content()

        # Step 5: Visualizations
        if self.config.save_figures:
            logger.info("\n[Step 5/6] Creating visualizations...")
            self.create_integration_map()
            self.create_amplitude_comparison_plot()

        # Step 6: Recommendations
        logger.info("\n[Step 6/6] Generating recommendations...")
        self.generate_recommendations()

        # Finalize
        end_time = datetime.now()
        self.results.processing_time_seconds = (end_time - start_time).total_seconds()

        # Save results
        results_file = Path(self.config.output_dir) / "integration_results.json"
        self.results.to_json(str(results_file))
        self.results.output_files['results_json'] = str(results_file)

        # Save config
        config_file = Path(self.config.output_dir) / "integration_config_used.json"
        self.config.to_json(str(config_file))

        logger.info("\n" + "=" * 80)
        logger.info("2D-3D INTEGRATION COMPLETE")
        logger.info(f"  Intersecting lines: {self.results.n_intersecting_lines}")
        logger.info(f"  Results: {results_file}")
        logger.info(f"  Time: {self.results.processing_time_seconds:.1f}s")
        logger.info("=" * 80)

        # Print recommendations
        if self.results.recommendations:
            logger.info("\nRecommendations:")
            for rec in self.results.recommendations:
                logger.info(f"  - {rec}")

        return self.results


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="2D-3D Seismic Integration Framework")
    parser.add_argument("-c", "--config", help="Configuration JSON file")
    parser.add_argument("--create-config", metavar="FILE", help="Generate default config")
    parser.add_argument("--survey-overlap", action="store_true", help="Only find overlapping lines")

    args = parser.parse_args()

    if args.create_config:
        config = Integration2D3DConfig()
        config.to_json(args.create_config)
        print(f"Created config: {args.create_config}")
        return

    if args.config:
        config = Integration2D3DConfig.from_json(args.config)
    else:
        config = Integration2D3DConfig()

    integration = Integration2D3D(config)

    if args.survey_overlap:
        integration.extract_3d_geometry()
        integration.load_2d_inventory()
        lines = integration.find_intersecting_lines()
        print(f"\nFound {len(lines)} intersecting lines:")
        for line in lines[:10]:
            print(f"  - {line['line_name']}: {line['intersection_length_m']:.0f}m intersection")
    else:
        results = integration.run()


if __name__ == "__main__":
    main()
