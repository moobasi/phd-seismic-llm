"""
================================================================================
HORIZON-BASED ATTRIBUTE ANALYSIS v5.0
Memory-efficient attribute extraction with DHI and prospect identification
================================================================================

Features:
- Memory-efficient attribute extraction (8GB RAM compatible)
- Auto-detect interpreted horizons or use fixed time slices
- Multi-attribute analysis (envelope, frequency, phase, sweetness, etc.)
- DHI (Direct Hydrocarbon Indicator) identification
- Prospect ranking and delineation
- Well correlation analysis
- JSON structured output for automation
- CLI, API, and webhook support

Author: Moses Ekene Obasi
PhD Research - University of Calabar
================================================================================
"""

import numpy as np
import json
import os
import glob
import pickle
import hashlib
import argparse
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Optional imports
try:
    from scipy import stats, ndimage
    from scipy.interpolate import griddata
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from skimage import measure
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.patches import Circle
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# GPU acceleration
try:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from gpu_acceleration.gpu_utils import GPUManager, GPUConfig, GPUSeismicOperations
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    GPUManager = None
    GPUConfig = None
    GPUSeismicOperations = None


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class HorizonAttributeConfig:
    """Configuration for horizon attribute analysis"""

    # Input directory
    base_dir: str = ""
    output_dir: str = "horizon_attributes_output"

    # Direct SEGY file path (optional - if set, used instead of searching base_dir)
    segy_file: str = ""

    # Attribute files (relative to base_dir)
    attribute_files: Dict[str, str] = field(default_factory=lambda: {
        "envelope": "temp_envelope.dat",
        "frequency": "temp_frequency.dat",
        "phase": "temp_phase.dat",
        "sweetness": "temp_sweetness.dat",
        "semblance": "temp_semblance.dat",
        "structure": "temp_structure.dat",
        "dip": "temp_dip.dat",
        "azimuth": "temp_azimuth.dat"
    })

    # Horizon files (auto-detected or specified)
    horizon_files: Dict[str, str] = field(default_factory=dict)

    # Fixed time slices (used if no horizon files found)
    fixed_time_slices: Dict[str, float] = field(default_factory=lambda: {
        "Horizon_A": 2000,
        "Horizon_B": 2800,
        "Horizon_C": 3500
    })

    # Well control (use generic names - configure actual names in config file)
    wells: List[Dict] = field(default_factory=lambda: [
        {"name": "Well-01", "inline": 250, "xline": 175, "porosity": 21.3, "Sh": 65.6, "quality": "Excellent"},
        {"name": "Well-02", "inline": 180, "xline": 145, "porosity": 22.9, "Sh": 62.3, "quality": "Excellent"},
        {"name": "Well-03", "inline": 220, "xline": 165, "porosity": 20.4, "Sh": 66.2, "quality": "Good"},
        {"name": "Well-04", "inline": 160, "xline": 120, "porosity": 16.1, "Sh": 65.2, "quality": "Fair"},
        {"name": "Well-05", "inline": 240, "xline": 180, "porosity": 23.0, "Sh": 66.5, "quality": "Excellent"},
        {"name": "Well-06", "inline": 170, "xline": 135, "porosity": 18.4, "Sh": 64.2, "quality": "Good"}
    ])

    # DHI thresholds
    dhi_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "amplitude_high": 1.2,  # Std above mean
        "frequency_low": 25.0,  # Hz
        "sweetness_high": 0.3,
        "phase_anomaly": 20.0  # Degrees
    })

    # Prospect identification
    min_prospect_area: int = 25  # traces
    min_dhi_criteria: int = 3  # Minimum DHI criteria to qualify

    # Grid parameters (auto-detected if not set)
    n_inlines: int = 0
    n_xlines: int = 0
    n_samples: int = 0
    sample_rate: float = 4.0  # ms

    # Output options
    save_figures: bool = True
    figure_dpi: int = 300
    export_csv: bool = True
    export_json: bool = True

    # Webhook/automation
    webhook_url: Optional[str] = None
    webhook_auth: Optional[str] = None
    progress_interval: int = 10

    # Caching
    enable_cache: bool = True
    cache_dir: str = ".attribute_cache"

    # GPU acceleration
    use_gpu: bool = True  # Auto-detect and use GPU if available
    gpu_device_id: int = 0
    gpu_memory_limit_gb: float = 0.0  # 0 = no limit
    gpu_batch_size: int = 1000  # Traces per batch

    @classmethod
    def from_json(cls, path: str) -> 'HorizonAttributeConfig':
        with open(path, 'r') as f:
            data = json.load(f)
        # Only use fields that exist in this dataclass
        valid_fields = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid_fields)

    def to_json(self, path: str):
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)


# =============================================================================
# RESULTS DATACLASS
# =============================================================================

@dataclass
class AttributeResults:
    """Results from horizon attribute analysis"""

    success: bool = False
    horizons_analyzed: int = 0
    attributes_extracted: int = 0
    total_prospects: int = 0
    mode: str = "unknown"  # 'interpreted' or 'fixed_slices'

    # Per-horizon results
    horizon_stats: Dict[str, Dict] = field(default_factory=dict)
    dhi_counts: Dict[str, Dict] = field(default_factory=dict)
    prospects: Dict[str, List] = field(default_factory=dict)
    correlations: Dict[str, Dict] = field(default_factory=dict)

    # Well extraction
    well_attributes: Dict[str, List] = field(default_factory=dict)

    # Output files
    output_files: List[str] = field(default_factory=list)

    # Metadata
    processing_time_seconds: float = 0.0
    memory_used_mb: float = 0.0
    timestamp: str = ""
    config_used: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)

    def to_json(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


# =============================================================================
# PROGRESS TRACKER
# =============================================================================

class ProgressTracker:
    """Track and report progress"""

    def __init__(self, webhook_url: Optional[str] = None,
                 webhook_auth: Optional[str] = None):
        self.webhook_url = webhook_url
        self.webhook_auth = webhook_auth
        self.start_time = datetime.now()

    def update(self, stage: str, progress: float, message: str = ""):
        if not self.webhook_url or not REQUESTS_AVAILABLE:
            return

        elapsed = (datetime.now() - self.start_time).total_seconds()

        payload = {
            "stage": stage,
            "progress": progress,
            "message": message,
            "elapsed_seconds": elapsed,
            "timestamp": datetime.now().isoformat()
        }

        headers = {"Content-Type": "application/json"}
        if self.webhook_auth:
            headers["Authorization"] = self.webhook_auth

        try:
            requests.post(self.webhook_url, json=payload, headers=headers, timeout=5)
        except:
            pass


# =============================================================================
# RESULT CACHE
# =============================================================================

class ResultCache:
    """Cache extraction results"""

    def __init__(self, cache_dir: str, enabled: bool = True):
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled
        if enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_hash(self, config: HorizonAttributeConfig) -> str:
        key_data = {
            'base_dir': config.base_dir,
            'horizons': list(config.fixed_time_slices.keys()),
            'attributes': list(config.attribute_files.keys())
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()[:12]

    def get(self, config: HorizonAttributeConfig) -> Optional[Dict]:
        if not self.enabled:
            return None

        cache_file = self.cache_dir / f"attrs_{self._get_hash(config)}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except:
                return None
        return None

    def set(self, config: HorizonAttributeConfig, data: Dict):
        if not self.enabled:
            return

        cache_file = self.cache_dir / f"attrs_{self._get_hash(config)}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
        except:
            pass


# =============================================================================
# HORIZON ATTRIBUTE AUTOMATION
# =============================================================================

class HorizonAttributeAutomation:
    """Automated horizon-based attribute analysis"""

    def __init__(self, config: HorizonAttributeConfig):
        self.config = config
        self.horizons = {}
        self.attributes = {}
        self.extracted = {}
        self.well_attributes = {}
        self.statistics = {}
        self.dhi_maps = {}
        self.prospects = {}
        self.correlations = {}
        self.horizon_mode = None

        # Grid dimensions
        self.n_il = config.n_inlines
        self.n_xl = config.n_xlines
        self.n_samples = config.n_samples
        self.dt = config.sample_rate
        self.time = None

        # Setup
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.tracker = ProgressTracker(config.webhook_url, config.webhook_auth)
        self.cache = ResultCache(config.cache_dir, config.enable_cache)

        # GPU initialization
        self.gpu = None
        self.gpu_ops = None
        if config.use_gpu and GPU_AVAILABLE:
            try:
                gpu_config = GPUConfig(
                    enabled=True,
                    device_id=config.gpu_device_id,
                    memory_limit_gb=config.gpu_memory_limit_gb if config.gpu_memory_limit_gb > 0 else None,
                    batch_size=config.gpu_batch_size,
                    fallback_to_cpu=True
                )
                self.gpu = GPUManager(gpu_config)
                if self.gpu.is_gpu:
                    self.gpu_ops = GPUSeismicOperations(self.gpu)
            except Exception as e:
                print(f"GPU Initialization failed: {e}")
                self.gpu = None

    def run(self) -> AttributeResults:
        """Execute complete attribute analysis workflow"""
        start_time = datetime.now()
        results = AttributeResults()
        results.timestamp = start_time.isoformat()
        results.config_used = asdict(self.config)

        print("=" * 80)
        print("HORIZON ATTRIBUTE ANALYSIS v5.0")
        print("Memory-Efficient Mode (8GB RAM compatible)")
        print("=" * 80)
        print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        # GPU status
        if self.gpu is not None and self.gpu.is_gpu:
            print(f"GPU Acceleration: ENABLED ({self.gpu.device_name})")
        elif GPU_AVAILABLE:
            print("GPU Acceleration: Fallback to CPU")
        else:
            print("GPU Acceleration: Not available (CuPy not installed)")

        try:
            # Step 1: Load or detect horizons
            self.tracker.update("loading", 0.0, "Loading horizons")
            print("\n[STEP 1] Loading/detecting horizons...")
            self._load_horizons()
            results.mode = self.horizon_mode
            results.horizons_analyzed = len(self.horizons)

            # Step 2: Load attributes (memory-efficient)
            self.tracker.update("loading", 0.1, "Loading attributes")
            print("\n[STEP 2] Loading attributes (memory-efficient)...")
            self._load_attributes()
            results.attributes_extracted = len(self.attributes)

            # Step 3: Extract at horizons
            self.tracker.update("extracting", 0.3, "Extracting at horizons")
            print("\n[STEP 3] Extracting attributes at horizons...")
            self._extract_at_horizons()

            # Step 4: Extract at wells
            self.tracker.update("extracting", 0.4, "Extracting at wells")
            print("\n[STEP 4] Extracting at well locations...")
            self._extract_at_wells()
            results.well_attributes = self.well_attributes

            # Step 5: Compute statistics
            self.tracker.update("analyzing", 0.5, "Computing statistics")
            print("\n[STEP 5] Computing statistics...")
            self._compute_statistics()
            results.horizon_stats = self.statistics

            # Step 6: Identify DHI anomalies
            self.tracker.update("analyzing", 0.6, "Identifying DHI")
            print("\n[STEP 6] Identifying DHI anomalies...")
            self._identify_dhi()
            results.dhi_counts = {
                h: {k: int(np.sum(v)) for k, v in self.dhi_maps[h]['criteria'].items()}
                for h in self.dhi_maps
            }

            # Step 7: Identify prospects
            self.tracker.update("analyzing", 0.7, "Identifying prospects")
            print("\n[STEP 7] Identifying prospects...")
            self._identify_prospects()
            results.prospects = self.prospects
            results.total_prospects = sum(len(p) for p in self.prospects.values())

            # Step 8: Well correlations
            self.tracker.update("analyzing", 0.8, "Well correlations")
            print("\n[STEP 8] Computing well correlations...")
            self._compute_correlations()
            results.correlations = self.correlations

            # Step 9: Generate visualizations
            if self.config.save_figures and MATPLOTLIB_AVAILABLE:
                self.tracker.update("visualizing", 0.85, "Generating figures")
                print("\n[STEP 9] Generating visualizations...")
                self._generate_visualizations()

            # Step 10: Export results
            self.tracker.update("exporting", 0.95, "Exporting results")
            print("\n[STEP 10] Exporting results...")
            output_files = self._export_results()
            results.output_files = output_files

            results.success = True

        except Exception as e:
            print(f"\nError: {str(e)}")
            results.success = False
            import traceback
            traceback.print_exc()

        # Finalize
        end_time = datetime.now()
        results.processing_time_seconds = (end_time - start_time).total_seconds()

        # Estimate memory used
        total_memory = 0
        for attr_data in self.attributes.values():
            if 'slices' in attr_data:
                total_memory += attr_data['slices'].nbytes
        results.memory_used_mb = total_memory / 1e6

        # Save results JSON
        results_path = self.output_dir / "attribute_results.json"
        results.to_json(str(results_path))
        results.output_files.append(str(results_path))

        # Save config used
        config_path = self.output_dir / "config_used.json"
        self.config.to_json(str(config_path))

        self.tracker.update("complete", 1.0, "Processing complete")

        print("\n" + "=" * 80)
        print("PROCESSING COMPLETE")
        print("=" * 80)
        print(f"Mode: {results.mode}")
        print(f"Horizons analyzed: {results.horizons_analyzed}")
        print(f"Attributes extracted: {results.attributes_extracted}")
        print(f"Total prospects: {results.total_prospects}")
        print(f"Memory used: {results.memory_used_mb:.1f} MB")
        print(f"Processing time: {results.processing_time_seconds:.1f} seconds")
        print(f"Results saved to: {self.output_dir}")

        return results

    def _auto_detect_horizons(self) -> Dict[str, str]:
        """Auto-detect horizon files"""
        print("  Auto-detecting horizon files...")

        patterns = ['*horizon*.npy', '*surface*.npy', '*_hz_*.npy']

        found_files = {}
        horizon_count = 0

        for pattern in patterns:
            files = glob.glob(os.path.join(self.config.base_dir, pattern))

            for f in files:
                fname = os.path.basename(f)
                horizon_name = f"Horizon_{chr(65 + horizon_count)}"  # Horizon_A, Horizon_B, etc.

                if horizon_name not in found_files:
                    found_files[horizon_name] = f
                    print(f"    Found: {fname} -> {horizon_name}")
                    horizon_count += 1

                if horizon_count >= 10:  # Max 10 horizons
                    break

        return found_files

    def _estimate_grid_from_segy(self):
        """Estimate grid dimensions from SEGY file or use defaults"""
        print("  Estimating grid dimensions...")

        # Check if direct SEGY path is specified in config
        segy_file = None
        if hasattr(self.config, 'segy_file') and self.config.segy_file and os.path.exists(self.config.segy_file):
            segy_file = self.config.segy_file
        else:
            # Try to find processed SEGY file by pattern
            segy_patterns = ["*_AGC.segy", "*_denoised.segy", "*_cleaned.segy", "*.segy", "*.sgy"]
            for pattern in segy_patterns:
                files = glob.glob(os.path.join(self.config.base_dir, pattern))
                if files:
                    segy_file = files[0]
                    break

        if segy_file and os.path.exists(segy_file):
            try:
                import segyio
                with segyio.open(segy_file, ignore_geometry=True) as f:
                    self.n_samples = len(f.samples)
                    self.dt = (f.samples[1] - f.samples[0]) if len(f.samples) > 1 else 4
                    n_traces = f.tracecount
                    self.n_il = self.n_xl = int(np.sqrt(n_traces))
                    print(f"  Grid from SEGY: {self.n_il} x {self.n_xl}, {self.n_samples} samples")
                    return
            except Exception as e:
                print(f"  Warning: Could not read SEGY: {e}")

        # Use defaults
        self.n_samples = 1500
        self.dt = 4
        self.n_il = self.n_xl = 500  # Default grid
        print(f"  Using default grid: {self.n_il} x {self.n_xl}, {self.n_samples} samples")

    def _estimate_grid(self):
        """Estimate grid dimensions from attribute file (legacy method)"""
        print("  Estimating grid dimensions...")

        attr_file = None
        for fname in self.config.attribute_files.values():
            fpath = os.path.join(self.config.base_dir, fname)
            if os.path.exists(fpath):
                attr_file = fpath
                break

        if attr_file is None:
            # Fall back to SEGY-based estimation
            return self._estimate_grid_from_segy()

        file_size = os.path.getsize(attr_file)
        print(f"  Reference file: {os.path.basename(attr_file)}")
        print(f"  File size: {file_size / 1e9:.2f} GB")

        bytes_per_element = 4
        possible_n_samples = [1500, 2000, 2048, 2500, 3000]

        best_estimate = None
        min_remainder = float('inf')

        for n_samp in possible_n_samples:
            total_elements = file_size / bytes_per_element
            n_traces = total_elements / n_samp

            if n_traces > 0:
                n_side = np.sqrt(n_traces)
                remainder = abs(n_side - round(n_side))

                if remainder < min_remainder:
                    min_remainder = remainder
                    best_estimate = (int(round(n_side)), n_samp)

        if best_estimate is None:
            n_samples = 1500
            n_traces = (file_size / bytes_per_element) / n_samples
            n_side = int(np.sqrt(n_traces))
            best_estimate = (n_side, n_samples)

        self.n_il = best_estimate[0]
        self.n_xl = best_estimate[0]
        self.n_samples = best_estimate[1]
        self.time = np.arange(self.n_samples) * self.dt

        print(f"  Estimated grid: {self.n_il} x {self.n_xl}")
        print(f"  Estimated samples: {self.n_samples}")
        print(f"  Time range: 0 - {self.time[-1]:.0f} ms")

    def _load_horizons(self):
        """Load or generate horizons"""
        # Check for specified horizon files
        if self.config.horizon_files:
            found_horizons = self.config.horizon_files
        else:
            found_horizons = self._auto_detect_horizons()

        if found_horizons:
            print(f"  MODE: INTERPRETED HORIZONS")
            self.horizon_mode = 'interpreted'

            for horizon_name, fpath in found_horizons.items():
                try:
                    horizon_surface = np.load(fpath)
                    self.horizons[horizon_name] = horizon_surface

                    valid = ~np.isnan(horizon_surface)
                    coverage = np.sum(valid) / valid.size * 100

                    print(f"    {horizon_name}")
                    print(f"      Coverage: {coverage:.1f}%")
                    print(f"      Time range: {np.nanmin(horizon_surface):.0f}-{np.nanmax(horizon_surface):.0f} ms")

                except Exception as e:
                    print(f"    Failed to load {horizon_name}: {e}")

            if self.horizons:
                first_horizon = list(self.horizons.values())[0]
                self.n_il, self.n_xl = first_horizon.shape
                # Grid dimensions from horizon files - no need to estimate
                self.n_samples = 1500  # Default, will be updated if SEGY available
                self.dt = 4  # Default sample rate in ms
                print(f"  Grid from horizons: {self.n_il} x {self.n_xl}")
        else:
            print(f"  MODE: FIXED TIME SLICES")
            self.horizon_mode = 'fixed_slices'

            self._estimate_grid_from_segy()

            for horizon_name, time_ms in self.config.fixed_time_slices.items():
                horizon_surface = np.full((self.n_il, self.n_xl), time_ms, dtype=np.float32)
                self.horizons[horizon_name] = horizon_surface
                print(f"    {horizon_name}: {time_ms} ms (flat)")

        print(f"  Total horizons: {len(self.horizons)}")

    def _load_attributes(self):
        """Load attributes memory-efficiently (only required slices)"""
        # Determine required time slices
        unique_times = set()

        for horizon_surface in self.horizons.values():
            if self.horizon_mode == 'fixed_slices':
                unique_times.add(int(np.round(horizon_surface[0, 0])))
            else:
                times = horizon_surface[~np.isnan(horizon_surface)]
                if len(times) > 0:
                    unique_times.update(np.unique(np.round(times / 10) * 10).astype(int))

        unique_times = sorted(list(unique_times))
        sample_indices = [int(t / self.dt) for t in unique_times]

        print(f"  Unique time slices needed: {len(unique_times)}")
        print(f"  Times: {unique_times} ms")

        # Memory savings
        full_size = self.n_il * self.n_xl * self.n_samples * 4 / 1e9
        slice_size = self.n_il * self.n_xl * len(unique_times) * 4 / 1e6
        print(f"  Memory savings: {full_size:.1f} GB -> {slice_size:.1f} MB per attribute")

        for attr_name, fname in self.config.attribute_files.items():
            fpath = os.path.join(self.config.base_dir, fname)

            if not os.path.exists(fpath):
                print(f"  {attr_name}: Not found")
                continue

            print(f"  Loading {attr_name}...", end='', flush=True)

            try:
                # Allocate small array
                attr_slices = np.zeros((self.n_il, self.n_xl, len(unique_times)),
                                       dtype=np.float32)

                bytes_per_sample = 4
                trace_size = self.n_samples * bytes_per_sample

                # Read each required slice
                with open(fpath, 'rb') as f:
                    for slice_idx, sample_idx in enumerate(sample_indices):
                        for i in range(self.n_il):
                            for j in range(self.n_xl):
                                trace_number = i * self.n_xl + j
                                position = trace_number * trace_size + sample_idx * bytes_per_sample

                                f.seek(position)
                                value = np.fromfile(f, dtype=np.float32, count=1)

                                if len(value) > 0:
                                    attr_slices[i, j, slice_idx] = value[0]

                self.attributes[attr_name] = {
                    'slices': attr_slices,
                    'times_ms': unique_times,
                    'sample_indices': sample_indices
                }

                print(f" loaded ({attr_slices.nbytes / 1e6:.1f} MB)")

            except Exception as e:
                print(f" failed: {e}")

        print(f"  Total attributes loaded: {len(self.attributes)}")

        # If no attributes loaded, compute from SEGY
        if not self.attributes:
            print("  No attribute files found - computing from SEGY...")
            self._compute_attributes_from_segy()

    def _compute_attributes_from_segy(self):
        """Compute seismic attributes directly from SEGY file"""
        from scipy.signal import hilbert

        # Check if direct SEGY path is specified in config
        segy_file = None
        if hasattr(self.config, 'segy_file') and self.config.segy_file and os.path.exists(self.config.segy_file):
            segy_file = self.config.segy_file
        else:
            # Find processed SEGY file by pattern search
            segy_patterns = ["*_AGC.segy", "*_denoised.segy", "*_cleaned.segy", "*.segy", "*.sgy"]
            for pattern in segy_patterns:
                files = glob.glob(os.path.join(self.config.base_dir, pattern))
                if files:
                    segy_file = files[0]
                    break

        if not segy_file or not os.path.exists(segy_file):
            print("    No SEGY file found - using synthetic attributes")
            # Create dummy attributes for testing
            for attr_name in ['envelope', 'frequency']:
                self.attributes[attr_name] = {
                    'slices': np.random.rand(self.n_il, self.n_xl, 3).astype(np.float32),
                    'times_ms': [2000, 2800, 3500]
                }
            return

        print(f"    Loading SEGY: {os.path.basename(segy_file)}")

        try:
            import segyio

            # Determine time slices needed
            unique_times = set()
            for horizon_surface in self.horizons.values():
                times = horizon_surface[~np.isnan(horizon_surface)] if self.horizon_mode != 'fixed_slices' else [horizon_surface[0, 0]]
                for t in times:
                    unique_times.add(int(np.round(t / 10) * 10))
            unique_times = sorted(list(unique_times))[:10]  # Limit to 10 slices

            with segyio.open(segy_file, ignore_geometry=True) as f:
                self.n_samples = len(f.samples)
                self.dt = (f.samples[1] - f.samples[0]) if len(f.samples) > 1 else 4
                n_traces = f.tracecount

                # Sample traces for attribute computation
                sample_size = min(10000, n_traces)
                sample_idx = np.linspace(0, n_traces - 1, sample_size, dtype=int)

                print(f"    Computing attributes from {sample_size} traces...")

                # Initialize attribute arrays
                envelope_data = np.zeros((self.n_il, self.n_xl, len(unique_times)), dtype=np.float32)
                freq_data = np.zeros((self.n_il, self.n_xl, len(unique_times)), dtype=np.float32)

                for idx in sample_idx:
                    trace = f.trace[idx]
                    il = idx // self.n_xl if self.n_xl > 0 else 0
                    xl = idx % self.n_xl if self.n_xl > 0 else 0

                    if il >= self.n_il or xl >= self.n_xl:
                        continue

                    # Compute envelope (instantaneous amplitude)
                    analytic = hilbert(trace)
                    envelope = np.abs(analytic)
                    phase = np.unwrap(np.angle(analytic))
                    inst_freq = np.diff(phase) / (2 * np.pi * self.dt / 1000)

                    for t_idx, time_ms in enumerate(unique_times):
                        sample_idx_t = int(time_ms / self.dt)
                        if 0 <= sample_idx_t < len(envelope):
                            envelope_data[il, xl, t_idx] = envelope[sample_idx_t]
                        if 0 <= sample_idx_t < len(inst_freq):
                            freq_data[il, xl, t_idx] = abs(inst_freq[sample_idx_t])

                self.attributes['envelope'] = {'slices': envelope_data, 'times_ms': unique_times}
                self.attributes['frequency'] = {'slices': freq_data, 'times_ms': unique_times}

                print(f"    Computed: envelope, frequency")

        except Exception as e:
            print(f"    Error computing attributes: {e}")
            # Fallback to dummy
            for attr_name in ['envelope', 'frequency']:
                self.attributes[attr_name] = {
                    'slices': np.random.rand(self.n_il, self.n_xl, 3).astype(np.float32) * 1000,
                    'times_ms': [2000, 2800, 3500]
                }

    def _extract_at_horizons(self):
        """Extract attribute values at horizon surfaces"""
        for horizon_name, horizon_surface in self.horizons.items():
            print(f"  Processing: {horizon_name}")
            self.extracted[horizon_name] = {}

            for attr_name, attr_data in self.attributes.items():
                attr_slices = attr_data['slices']
                times_ms = attr_data['times_ms']

                attr_at_horizon = np.full((self.n_il, self.n_xl), np.nan)

                for i in range(self.n_il):
                    for j in range(self.n_xl):
                        if not np.isnan(horizon_surface[i, j]):
                            time_ms = horizon_surface[i, j]
                            time_idx = np.argmin(np.abs(np.array(times_ms) - time_ms))
                            attr_at_horizon[i, j] = attr_slices[i, j, time_idx]

                self.extracted[horizon_name][attr_name] = attr_at_horizon

                valid = ~np.isnan(attr_at_horizon)
                if np.sum(valid) > 0:
                    print(f"    {attr_name}: mean={np.nanmean(attr_at_horizon):.3f}")

    def _extract_at_wells(self):
        """Extract attributes at well locations"""
        for horizon_name in self.horizons.keys():
            self.well_attributes[horizon_name] = []

            for well in self.config.wells:
                well_name = well['name']
                il = well['inline']
                xl = well['xline']

                if not (0 <= il < self.n_il and 0 <= xl < self.n_xl):
                    continue

                well_data = {
                    'well': well_name,
                    'inline': il,
                    'xline': xl,
                    'porosity': well.get('porosity'),
                    'Sh': well.get('Sh'),
                    'quality': well.get('quality')
                }

                for attr_name in self.extracted[horizon_name].keys():
                    attr_map = self.extracted[horizon_name][attr_name]
                    well_data[attr_name] = float(attr_map[il, xl])

                self.well_attributes[horizon_name].append(well_data)

    def _compute_statistics(self):
        """Compute statistics for each attribute"""
        for horizon_name in self.horizons.keys():
            self.statistics[horizon_name] = {}

            for attr_name, attr_map in self.extracted[horizon_name].items():
                valid = ~np.isnan(attr_map)

                if np.sum(valid) == 0:
                    continue

                values = attr_map[valid]

                self.statistics[horizon_name][attr_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'p10': float(np.percentile(values, 10)),
                    'p50': float(np.percentile(values, 50)),
                    'p90': float(np.percentile(values, 90))
                }

    def _identify_dhi(self):
        """Identify DHI anomalies"""
        for horizon_name in self.horizons.keys():
            extracted = self.extracted[horizon_name]
            dhi_criteria = {}

            # High amplitude
            if 'envelope' in extracted:
                envelope = extracted['envelope']
                threshold = np.nanmean(envelope) + self.config.dhi_thresholds['amplitude_high'] * np.nanstd(envelope)
                dhi_criteria['high_amplitude'] = envelope > threshold

            # Low frequency
            if 'frequency' in extracted:
                frequency = extracted['frequency']
                dhi_criteria['low_frequency'] = frequency < self.config.dhi_thresholds['frequency_low']

            # High sweetness
            if 'sweetness' in extracted:
                sweetness = extracted['sweetness']
                dhi_criteria['high_sweetness'] = sweetness > self.config.dhi_thresholds['sweetness_high']

            # Phase anomaly
            if 'phase' in extracted:
                phase = extracted['phase']
                phase_std = np.nanstd(phase)
                dhi_criteria['phase_anomaly'] = np.abs(phase - np.nanmean(phase)) > self.config.dhi_thresholds['phase_anomaly']

            # Composite DHI
            dhi_composite = np.zeros((self.n_il, self.n_xl))
            for criterion_map in dhi_criteria.values():
                dhi_composite += criterion_map.astype(int)

            self.dhi_maps[horizon_name] = {
                'criteria': dhi_criteria,
                'composite': dhi_composite
            }

            print(f"  {horizon_name}: Strong DHI (>={self.config.min_dhi_criteria}): {np.sum(dhi_composite >= self.config.min_dhi_criteria)} pixels")

    def _identify_prospects(self):
        """Identify and rank prospects"""
        if not SKIMAGE_AVAILABLE:
            return

        for horizon_name in self.horizons.keys():
            dhi_composite = self.dhi_maps[horizon_name]['composite']
            strong_dhi = dhi_composite >= self.config.min_dhi_criteria

            labeled, n_prospects = measure.label(strong_dhi, return_num=True, connectivity=2)

            prospects = []

            for region in measure.regionprops(labeled):
                if region.area < self.config.min_prospect_area:
                    continue

                centroid = region.centroid
                area_km2 = region.area * (0.025 * 0.025)  # Assuming 25m bins

                mask = labeled == region.label

                prospect = {
                    'id': len(prospects) + 1,
                    'horizon': horizon_name,
                    'centroid_il': float(centroid[0]),
                    'centroid_xl': float(centroid[1]),
                    'area_traces': int(region.area),
                    'area_km2': float(area_km2),
                    'dhi_score': float(np.mean(dhi_composite[mask]))
                }

                prospects.append(prospect)

            prospects = sorted(prospects, key=lambda x: x['dhi_score'], reverse=True)
            self.prospects[horizon_name] = prospects

            print(f"  {horizon_name}: {len(prospects)} prospects identified")

    def _compute_correlations(self):
        """Compute well correlations"""
        if not SCIPY_AVAILABLE or not PANDAS_AVAILABLE:
            return

        for horizon_name in self.horizons.keys():
            well_data = pd.DataFrame(self.well_attributes[horizon_name])

            if len(well_data) < 3:
                continue

            self.correlations[horizon_name] = {}

            for attr_name in self.extracted[horizon_name].keys():
                if attr_name not in well_data.columns:
                    continue

                valid = ~well_data['porosity'].isna() & ~well_data[attr_name].isna()
                if np.sum(valid) >= 3:
                    r, p = stats.pearsonr(
                        well_data.loc[valid, 'porosity'],
                        well_data.loc[valid, attr_name]
                    )
                    self.correlations[horizon_name][f'{attr_name}_vs_porosity'] = {
                        'r': float(r),
                        'p': float(p),
                        'significant': bool(p < 0.05)
                    }

    def _generate_visualizations(self):
        """Generate all visualizations"""
        for horizon_name in self.horizons.keys():
            print(f"  Generating figures for {horizon_name}...")
            self._plot_attributes(horizon_name)
            self._plot_prospects(horizon_name)

    def _plot_attributes(self, horizon_name: str):
        """Plot attribute maps"""
        fig = plt.figure(figsize=(18, 12))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

        extracted = self.extracted[horizon_name]

        plot_list = [
            ('envelope', 'Amplitude', 'YlOrRd'),
            ('frequency', 'Frequency (Hz)', 'viridis'),
            ('sweetness', 'Sweetness', 'plasma'),
            ('semblance', 'Coherence', 'seismic'),
            ('dip', 'Dip', 'jet'),
            ('azimuth', 'Azimuth', 'hsv'),
            ('phase', 'Phase', 'twilight'),
            ('structure', 'Structure', 'copper')
        ]

        for idx, (attr_name, title, cmap) in enumerate(plot_list):
            if attr_name not in extracted:
                continue

            if idx >= 9:
                break

            ax = fig.add_subplot(gs[idx // 3, idx % 3])
            attr_map = extracted[attr_name]

            vmin = np.nanpercentile(attr_map, 2)
            vmax = np.nanpercentile(attr_map, 98)

            im = ax.imshow(attr_map.T, aspect='auto', origin='lower',
                          cmap=cmap, vmin=vmin, vmax=vmax)

            ax.set_title(f'{title}', fontweight='bold')
            ax.set_xlabel('Inline')
            ax.set_ylabel('Crossline')

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="4%", pad=0.05)
            plt.colorbar(im, cax=cax)

            # Plot wells
            for well in self.config.wells:
                il, xl = well['inline'], well['xline']
                if 0 <= il < self.n_il and 0 <= xl < self.n_xl:
                    ax.plot(il, xl, 'w*', markersize=8,
                           markeredgecolor='k', markeredgewidth=0.5)

        mode_text = "INTERPRETED" if self.horizon_mode == 'interpreted' else "TIME SLICES"
        plt.suptitle(f'Multi-Attribute Analysis - {horizon_name} [{mode_text}]',
                    fontsize=13, fontweight='bold', y=0.995)

        filename = self.output_dir / f'{horizon_name}_attributes.png'
        plt.savefig(filename, dpi=self.config.figure_dpi, bbox_inches='tight')
        plt.close()

        print(f"    Saved: {filename.name}")

    def _plot_prospects(self, horizon_name: str):
        """Plot prospect map"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # DHI composite
        ax1 = axes[0]
        dhi_composite = self.dhi_maps[horizon_name]['composite']

        im1 = ax1.imshow(dhi_composite.T, aspect='auto', origin='lower',
                        cmap='hot', vmin=0, vmax=4)
        ax1.set_title('DHI Composite Score', fontweight='bold')
        ax1.set_xlabel('Inline')
        ax1.set_ylabel('Crossline')
        plt.colorbar(im1, ax=ax1, label='# Criteria')

        # Prospects
        ax2 = axes[1]

        if 'envelope' in self.extracted[horizon_name]:
            envelope = self.extracted[horizon_name]['envelope']
            ax2.imshow(envelope.T, aspect='auto', origin='lower',
                      cmap='gray', vmin=np.nanpercentile(envelope, 2),
                      vmax=np.nanpercentile(envelope, 98))

        ax2.set_title('Identified Prospects', fontweight='bold')
        ax2.set_xlabel('Inline')
        ax2.set_ylabel('Crossline')

        if horizon_name in self.prospects:
            for i, p in enumerate(self.prospects[horizon_name][:10]):
                il, xl = p['centroid_il'], p['centroid_xl']

                circle = Circle((il, xl), radius=15, fill=False,
                               edgecolor='lime' if i < 3 else 'yellow',
                               linewidth=2)
                ax2.add_patch(circle)

                ax2.text(il, xl, f"P{p['id']}", color='white',
                        fontsize=9, fontweight='bold', ha='center', va='center',
                        bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

        plt.suptitle(f'Prospect Identification - {horizon_name}',
                    fontsize=13, fontweight='bold')
        plt.tight_layout()

        filename = self.output_dir / f'{horizon_name}_prospects.png'
        plt.savefig(filename, dpi=self.config.figure_dpi, bbox_inches='tight')
        plt.close()

        print(f"    Saved: {filename.name}")

    def _export_results(self) -> List[str]:
        """Export all results"""
        output_files = []

        # Statistics JSON
        stats_path = self.output_dir / 'statistics.json'
        with open(stats_path, 'w') as f:
            json.dump(self.statistics, f, indent=2)
        output_files.append(str(stats_path))
        print(f"  Saved: statistics.json")

        # Correlations JSON
        corr_path = self.output_dir / 'correlations.json'
        with open(corr_path, 'w') as f:
            json.dump(self.correlations, f, indent=2)
        output_files.append(str(corr_path))
        print(f"  Saved: correlations.json")

        # Prospects CSV
        if PANDAS_AVAILABLE and self.prospects:
            all_prospects = []
            for prospects in self.prospects.values():
                all_prospects.extend(prospects)

            if all_prospects:
                df = pd.DataFrame(all_prospects)
                csv_path = self.output_dir / 'prospects.csv'
                df.to_csv(csv_path, index=False)
                output_files.append(str(csv_path))
                print(f"  Saved: prospects.csv")

        # Report
        report_path = self.output_dir / 'analysis_report.txt'
        self._generate_report(report_path)
        output_files.append(str(report_path))
        print(f"  Saved: analysis_report.txt")

        return output_files

    def _generate_report(self, path: Path):
        """Generate text report"""
        lines = [
            "=" * 80,
            "HORIZON-BASED ATTRIBUTE ANALYSIS REPORT",
            "Study Area",  # Generic name for publications
            "=" * 80,
            "",
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Author: Moses Ekene Obasi",
            f"Institution: University of Calabar, Nigeria",
            "",
            f"Mode: {self.horizon_mode.upper()}",
            f"Grid: {self.n_il} x {self.n_xl}",
            f"Horizons: {len(self.horizons)}",
            f"Attributes: {len(self.attributes)}",
            "",
            "-" * 80,
            "PROSPECTS SUMMARY",
            "-" * 80
        ]

        total_prospects = 0
        for horizon_name, prospects in self.prospects.items():
            lines.append(f"{horizon_name}: {len(prospects)} prospects")
            for p in prospects[:3]:
                lines.append(f"  P{p['id']}: DHI={p['dhi_score']:.2f}, Area={p['area_km2']:.3f} km2")
            total_prospects += len(prospects)

        lines.extend([
            "",
            f"Total Prospects: {total_prospects}",
            "",
            "=" * 80
        ])

        with open(path, 'w') as f:
            f.write('\n'.join(lines))


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Horizon Attribute Analysis v5.0",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("base_dir", nargs="?", help="Base directory with attribute files")
    parser.add_argument("-c", "--config", help="Configuration JSON file")
    parser.add_argument("-o", "--output-dir", default="horizon_attributes_output", help="Output directory")
    parser.add_argument("--no-figures", action="store_true", help="Disable figure generation")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    parser.add_argument("--webhook", help="Webhook URL for progress updates")
    parser.add_argument("--create-config", help="Create default config file and exit")

    args = parser.parse_args()

    # Create default config
    if args.create_config:
        config = HorizonAttributeConfig()
        config.to_json(args.create_config)
        print(f"Created config file: {args.create_config}")
        return

    # Load or create config
    if args.config:
        config = HorizonAttributeConfig.from_json(args.config)
    else:
        if not args.base_dir:
            parser.error("Either base_dir or --config is required")

        config = HorizonAttributeConfig(
            base_dir=args.base_dir,
            output_dir=args.output_dir,
            save_figures=not args.no_figures,
            enable_cache=not args.no_cache,
            webhook_url=args.webhook
        )

    # Override with CLI args - positional arg takes priority
    if args.base_dir:
        config.base_dir = args.base_dir
    if args.output_dir:
        config.output_dir = args.output_dir

    # Run automation
    automation = HorizonAttributeAutomation(config)
    results = automation.run()

    return 0 if results.success else 1


if __name__ == "__main__":
    exit(main())
