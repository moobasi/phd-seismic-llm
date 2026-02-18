"""
Seismic EDA Automation Framework
Version 5.0 - Production-Ready for Automated Workflows

Features:
- Configuration-driven (JSON/CLI)
- Structured JSON output for n8n/automation
- REST API endpoint support
- Webhook callbacks for progress
- Modular pipeline architecture
- Caching for efficiency
- Proper logging
- Error handling

Author: Moses Ekene Obasi
PhD Research - University of Calabar
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for automation
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, signal
from scipy.stats import anderson, shapiro, pearsonr
import warnings
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Any, Callable, Union
from enum import Enum
import json
import logging
import hashlib
import pickle
import time
import argparse
import sys
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading

# Conditional imports with fallbacks
try:
    import segyio
    SEGYIO_AVAILABLE = True
except ImportError:
    SEGYIO_AVAILABLE = False
    print("WARNING: segyio not installed. Install with: pip install segyio")

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterable, **kwargs):
        return iterable

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("WARNING: pandas not installed. Some features disabled.")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# GPU acceleration
try:
    import cupy as cp
    GPU_AVAILABLE = True
    # Warm up GPU
    _warmup = cp.zeros(100)
    del _warmup
    cp.cuda.Stream.null.synchronize()
except ImportError:
    GPU_AVAILABLE = False
    cp = None

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('SeismicEDA')


# ============================================================================
# CONFIGURATION CLASSES
# ============================================================================

class OutputFormat(Enum):
    """Output format options"""
    JSON = "json"
    CSV = "csv"
    BOTH = "both"


@dataclass
class EDAConfig:
    """Configuration for EDA analysis"""
    # Input
    segy_file: str = ""

    # Processing
    chunk_size: int = 10000
    sample_traces: int = 30000
    n_spatial_bins: int = 10
    traces_per_bin: int = 300
    frequency_sample_traces: int = 3000
    attribute_sample_traces: int = 10000

    # Velocity model
    avg_velocity: float = 2500.0  # m/s

    # GPU acceleration
    use_gpu: bool = True  # Auto-use GPU if available

    # Output
    output_dir: str = "eda_outputs"
    output_format: str = "both"  # json, csv, both
    save_figures: bool = True
    figure_dpi: int = 300
    figure_format: str = "png"

    # Automation
    webhook_url: Optional[str] = None
    webhook_auth: Optional[str] = None
    progress_interval: int = 10  # Report every N% progress

    # Caching
    enable_cache: bool = True
    cache_dir: str = ".eda_cache"

    # Analysis flags
    run_statistics: bool = True
    run_quality_checks: bool = True
    run_frequency_analysis: bool = True
    run_resolution_analysis: bool = True
    run_attribute_analysis: bool = True
    run_spatial_analysis: bool = True

    @classmethod
    def from_json(cls, json_path: str) -> 'EDAConfig':
        """Load config from JSON file, ignoring unknown fields"""
        with open(json_path, 'r') as f:
            data = json.load(f)
        # Only use fields that exist in this dataclass
        valid_fields = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid_fields)

    @classmethod
    def from_dict(cls, data: Dict) -> 'EDAConfig':
        """Load config from dictionary"""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)

    def save(self, path: str):
        """Save config to JSON"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class EDAResults:
    """Structured results container"""
    # Metadata
    timestamp: str = ""
    segy_file: str = ""
    processing_time_seconds: float = 0.0
    version: str = "5.0"

    # Survey info
    survey: Dict = field(default_factory=dict)

    # Statistics
    statistics: Dict = field(default_factory=dict)

    # Quality
    quality: Dict = field(default_factory=dict)

    # Spectral
    spectral: Dict = field(default_factory=dict)

    # Resolution
    resolution: Dict = field(default_factory=dict)

    # Attributes
    attributes: Dict = field(default_factory=dict)

    # Spatial patterns
    spatial: Dict = field(default_factory=dict)

    # Recommendations
    recommendations: List[str] = field(default_factory=list)

    # Warnings
    warnings: List[str] = field(default_factory=list)

    # Output files
    output_files: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)

    def to_json(self, path: str):
        """Save to JSON file"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def from_json(cls, path: str) -> 'EDAResults':
        """Load from JSON file"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(**data)


# ============================================================================
# PROGRESS TRACKING
# ============================================================================

class ProgressTracker:
    """Track and report progress for automation"""

    def __init__(self, config: EDAConfig, total_steps: int = 100):
        self.config = config
        self.total_steps = total_steps
        self.current_step = 0
        self.current_phase = ""
        self.start_time = time.time()
        self._lock = threading.Lock()
        self._last_reported = 0

    def update(self, step: int = None, phase: str = None, message: str = None):
        """Update progress"""
        with self._lock:
            if step is not None:
                self.current_step = step
            if phase is not None:
                self.current_phase = phase

            progress_pct = (self.current_step / self.total_steps) * 100

            # Report at intervals
            if progress_pct - self._last_reported >= self.config.progress_interval:
                self._last_reported = progress_pct
                self._report_progress(progress_pct, message)

    def _report_progress(self, progress_pct: float, message: str = None):
        """Report progress to webhook if configured"""
        elapsed = time.time() - self.start_time

        payload = {
            "progress": round(progress_pct, 1),
            "phase": self.current_phase,
            "message": message or f"Processing: {self.current_phase}",
            "elapsed_seconds": round(elapsed, 1),
            "timestamp": datetime.now().isoformat()
        }

        logger.info(f"Progress: {progress_pct:.1f}% - {self.current_phase}")

        if self.config.webhook_url and REQUESTS_AVAILABLE:
            try:
                headers = {"Content-Type": "application/json"}
                if self.config.webhook_auth:
                    headers["Authorization"] = self.config.webhook_auth

                requests.post(
                    self.config.webhook_url,
                    json=payload,
                    headers=headers,
                    timeout=5
                )
            except Exception as e:
                logger.warning(f"Webhook failed: {e}")


# ============================================================================
# CACHING
# ============================================================================

class ResultCache:
    """Cache intermediate results for efficiency"""

    def __init__(self, config: EDAConfig):
        self.config = config
        self.cache_dir = Path(config.cache_dir)
        if config.enable_cache:
            self.cache_dir.mkdir(exist_ok=True)

    def _get_key(self, segy_file: str, operation: str) -> str:
        """Generate cache key based on file hash and operation"""
        # Use file path + size + mtime as hash source
        file_path = Path(segy_file)
        if file_path.exists():
            stat = file_path.stat()
            hash_source = f"{segy_file}:{stat.st_size}:{stat.st_mtime}:{operation}"
        else:
            hash_source = f"{segy_file}:{operation}"

        return hashlib.md5(hash_source.encode()).hexdigest()

    def get(self, segy_file: str, operation: str) -> Optional[Any]:
        """Get cached result"""
        if not self.config.enable_cache:
            return None

        key = self._get_key(segy_file, operation)
        cache_file = self.cache_dir / f"{key}.pkl"

        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    logger.debug(f"Cache hit: {operation}")
                    return pickle.load(f)
            except Exception:
                return None
        return None

    def set(self, segy_file: str, operation: str, data: Any):
        """Cache result"""
        if not self.config.enable_cache:
            return

        key = self._get_key(segy_file, operation)
        cache_file = self.cache_dir / f"{key}.pkl"

        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            logger.debug(f"Cached: {operation}")
        except Exception as e:
            logger.warning(f"Cache write failed: {e}")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def format_number(num: float, sig_figs: int = 3) -> float:
    """Format number with appropriate significant figures"""
    if num == 0 or np.isnan(num) or np.isinf(num):
        return 0.0
    if abs(num) < 0.001 or abs(num) > 10000:
        return float(f"{num:.{sig_figs-1}e}")
    elif abs(num) >= 100:
        return round(num, 0)
    elif abs(num) >= 10:
        return round(num, 1)
    elif abs(num) >= 1:
        return round(num, 2)
    else:
        return round(num, 3)


def calculate_snr_coherence(trace: np.ndarray) -> float:
    """
    Calculate SNR using coherence-based method
    More appropriate for post-stack data than window-based approaches
    """
    if np.all(trace == 0):
        return -999.0

    # Autocorrelation approach
    autocorr = np.correlate(trace, trace, mode='full')
    autocorr = autocorr[len(autocorr)//2:]

    # Signal energy: zero-lag autocorrelation
    signal_energy = autocorr[0]

    # Noise energy: mean of autocorrelation tail
    noise_lag_start = min(50, len(autocorr)//2)
    noise_energy = np.mean(np.abs(autocorr[noise_lag_start:]))

    if noise_energy > 0:
        snr = signal_energy / noise_energy
        return 10 * np.log10(snr) if snr > 0 else -999.0
    return -999.0


# ============================================================================
# MAIN EDA CLASS
# ============================================================================

class SeismicEDAAutomation:
    """
    Production-ready Seismic EDA with automation support

    Features:
    - Configuration-driven processing
    - Structured JSON output
    - Webhook progress callbacks
    - Caching for efficiency
    - Modular analysis pipeline
    """

    def __init__(self, config: Union[EDAConfig, Dict, str]):
        """
        Initialize EDA processor

        Parameters:
        -----------
        config : EDAConfig, dict, or str (path to JSON config)
        """
        # Load configuration
        if isinstance(config, str):
            if config.endswith('.json'):
                self.config = EDAConfig.from_json(config)
            else:
                # Assume it's a SEGY path
                self.config = EDAConfig(segy_file=config)
        elif isinstance(config, dict):
            self.config = EDAConfig.from_dict(config)
        else:
            self.config = config

        # Validate
        if not SEGYIO_AVAILABLE:
            raise ImportError("segyio is required. Install with: pip install segyio")

        if not self.config.segy_file:
            raise ValueError("segy_file must be specified in config")

        # Setup
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.cache = ResultCache(self.config)
        self.progress = ProgressTracker(self.config, total_steps=100)
        self.results = EDAResults(
            timestamp=datetime.now().isoformat(),
            segy_file=self.config.segy_file
        )

        # File handle
        self.f = None

        # Survey properties (populated during analysis)
        self.n_traces = 0
        self.n_samples = 0
        self.sample_rate = 0.0
        self.time = None
        self.inline_range = (0, 0)
        self.xline_range = (0, 0)
        self.n_inlines = 0
        self.n_xlines = 0

    def __enter__(self):
        """Context manager entry"""
        self.f = segyio.open(self.config.segy_file, ignore_geometry=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self.f:
            self.f.close()

    # ========================================================================
    # ANALYSIS METHODS
    # ========================================================================

    def load_metadata(self) -> Dict:
        """Load survey metadata and geometry"""
        logger.info("Loading metadata...")
        self.progress.update(step=5, phase="Loading Metadata")

        # Check cache
        cached = self.cache.get(self.config.segy_file, "metadata")
        if cached:
            for key, val in cached.items():
                setattr(self, key, val)
            self.results.survey = cached.get('survey_info', {})
            return cached

        # Basic info
        self.n_traces = self.f.tracecount
        self.n_samples = self.f.samples.size
        self.sample_rate = segyio.tools.dt(self.f) / 1000  # ms
        self.time = self.f.samples

        # Try to get 3D geometry
        try:
            self.inline_range = (self.f.ilines[0], self.f.ilines[-1])
            self.xline_range = (self.f.xlines[0], self.f.xlines[-1])
            self.n_inlines = len(self.f.ilines)
            self.n_xlines = len(self.f.xlines)
        except:
            # Sample traces for geometry
            sample_size = min(30000, self.n_traces)
            sample_idx = np.linspace(0, self.n_traces-1, sample_size, dtype=int)

            inlines, xlines = [], []
            for idx in tqdm(sample_idx, desc="Sampling geometry"):
                header = self.f.header[int(idx)]
                inlines.append(header[segyio.TraceField.INLINE_3D])
                xlines.append(header[segyio.TraceField.CROSSLINE_3D])

            inlines = np.array(inlines)
            xlines = np.array(xlines)

            self.inline_range = (int(np.min(inlines)), int(np.max(inlines)))
            self.xline_range = (int(np.min(xlines)), int(np.max(xlines)))
            self.n_inlines = len(np.unique(inlines))
            self.n_xlines = len(np.unique(xlines))

        # Extract CDP coordinates from sample
        sample_size = min(1000, self.n_traces)
        sample_idx = np.linspace(0, self.n_traces-1, sample_size, dtype=int)

        cdp_x, cdp_y = [], []
        for idx in sample_idx:
            header = self.f.header[int(idx)]
            cdp_x.append(header[segyio.TraceField.CDP_X])
            cdp_y.append(header[segyio.TraceField.CDP_Y])

        cdp_x = np.array(cdp_x, dtype=np.float64)
        cdp_y = np.array(cdp_y, dtype=np.float64)

        # Apply coordinate scalar
        coord_scalar = self.f.header[0][segyio.TraceField.SourceGroupScalar]
        if coord_scalar < 0:
            coord_scalar = -1.0 / coord_scalar
        elif coord_scalar == 0:
            coord_scalar = 1.0

        cdp_x *= coord_scalar
        cdp_y *= coord_scalar

        # Calculate survey dimensions
        survey_x = float(np.max(cdp_x) - np.min(cdp_x))
        survey_y = float(np.max(cdp_y) - np.min(cdp_y))
        survey_area = (survey_x * survey_y) / 1e6  # km²

        bin_inline = survey_x / self.n_inlines if self.n_inlines > 1 else 25.0
        bin_xline = survey_y / self.n_xlines if self.n_xlines > 1 else 25.0

        expected_traces = self.n_inlines * self.n_xlines
        completeness = 100 * self.n_traces / expected_traces if expected_traces > 0 else 0

        # Depth estimates
        max_depth = (self.time[-1] / 1000) * self.config.avg_velocity / 2

        # Build survey info
        survey_info = {
            "total_traces": int(self.n_traces),
            "samples_per_trace": int(self.n_samples),
            "sample_interval_ms": float(self.sample_rate),
            "record_length_ms": float(self.time[-1]),
            "inline_range": [int(self.inline_range[0]), int(self.inline_range[1])],
            "xline_range": [int(self.xline_range[0]), int(self.xline_range[1])],
            "n_inlines": int(self.n_inlines),
            "n_xlines": int(self.n_xlines),
            "expected_traces": int(expected_traces),
            "completeness_pct": format_number(completeness),
            "cdp_x_range": [format_number(float(np.min(cdp_x))), format_number(float(np.max(cdp_x)))],
            "cdp_y_range": [format_number(float(np.min(cdp_y))), format_number(float(np.max(cdp_y)))],
            "survey_extent_x_m": format_number(survey_x),
            "survey_extent_y_m": format_number(survey_y),
            "survey_area_km2": format_number(survey_area),
            "bin_size_inline_m": format_number(bin_inline),
            "bin_size_xline_m": format_number(bin_xline),
            "nyquist_frequency_hz": format_number(1000 / (2 * self.sample_rate)),
            "avg_velocity_ms": float(self.config.avg_velocity),
            "max_depth_m": format_number(max_depth),
            "data_size_gb": format_number((self.n_traces * self.n_samples * 4) / (1024**3))
        }

        self.results.survey = survey_info

        # Cache
        cache_data = {
            'n_traces': self.n_traces,
            'n_samples': self.n_samples,
            'sample_rate': self.sample_rate,
            'time': self.time,
            'inline_range': self.inline_range,
            'xline_range': self.xline_range,
            'n_inlines': self.n_inlines,
            'n_xlines': self.n_xlines,
            'survey_info': survey_info
        }
        self.cache.set(self.config.segy_file, "metadata", cache_data)

        logger.info(f"Survey: {self.n_traces:,} traces, {survey_area:.1f} km²")
        return survey_info

    def stratified_sampling(self) -> np.ndarray:
        """Get stratified spatial sample of trace indices"""
        logger.info("Generating stratified sample...")

        # Check cache
        cached = self.cache.get(self.config.segy_file, "stratified_sample")
        if cached is not None:
            return cached

        n_bins = self.config.n_spatial_bins
        traces_per_bin = self.config.traces_per_bin

        inline_bins = np.linspace(self.inline_range[0], self.inline_range[1], n_bins + 1)
        xline_bins = np.linspace(self.xline_range[0], self.xline_range[1], n_bins + 1)

        sample_indices = []

        for i in range(n_bins):
            for j in range(n_bins):
                bin_traces = []
                check_size = min(5000, self.n_traces)
                check_idx = np.random.choice(self.n_traces, check_size, replace=False)

                for idx in check_idx:
                    header = self.f.header[int(idx)]
                    il = header[segyio.TraceField.INLINE_3D]
                    xl = header[segyio.TraceField.CROSSLINE_3D]

                    if (inline_bins[i] <= il < inline_bins[i+1] and
                        xline_bins[j] <= xl < xline_bins[j+1]):
                        bin_traces.append(idx)

                    if len(bin_traces) >= traces_per_bin:
                        break

                if bin_traces:
                    n_sample = min(traces_per_bin, len(bin_traces))
                    sample_indices.extend(np.random.choice(bin_traces, n_sample, replace=False))

        sample_indices = np.array(sample_indices, dtype=int)

        # Fallback to random sampling if stratified sampling fails
        if len(sample_indices) == 0:
            logger.warning("Stratified sampling returned 0 traces, falling back to random sampling")
            n_sample = min(1000, self.n_traces)
            sample_indices = np.random.choice(self.n_traces, n_sample, replace=False)

        self.cache.set(self.config.segy_file, "stratified_sample", sample_indices)

        logger.info(f"Stratified sample: {len(sample_indices):,} traces")
        return sample_indices

    def compute_statistics(self) -> Dict:
        """Compute amplitude statistics using Welford's streaming algorithm"""
        logger.info("Computing statistics...")
        self.progress.update(step=15, phase="Computing Statistics")

        if not self.config.run_statistics:
            return {}

        # Check cache
        cached = self.cache.get(self.config.segy_file, "statistics")
        if cached:
            self.results.statistics = cached
            return cached

        sample_idx = self.stratified_sampling()

        # Welford's streaming algorithm for stability
        n = 0
        mean = 0.0
        M2 = 0.0
        min_val = np.inf
        max_val = -np.inf
        n_zeros = 0

        all_amps = []  # For validation subset

        for idx in tqdm(sample_idx, desc="Statistics"):
            trace = self.f.trace[int(idx)]

            for x in trace:
                n += 1
                delta = x - mean
                mean += delta / n
                delta2 = x - mean
                M2 += delta * delta2

                min_val = min(min_val, x)
                max_val = max(max_val, x)

                if x == 0:
                    n_zeros += 1

            if len(all_amps) < 1000000:
                all_amps.extend(trace)

        variance = M2 / (n - 1) if n > 1 else 0
        std_dev = np.sqrt(variance)
        rms = np.sqrt(mean**2 + variance)

        # Validation on subset
        val_data = np.array(all_amps[:500000])
        val_skew = float(stats.skew(val_data)) if len(val_data) > 100 else 0
        val_kurt = float(stats.kurtosis(val_data)) if len(val_data) > 100 else 0

        # Normality tests (require minimum samples)
        test_data = val_data[:5000]
        if len(test_data) >= 20:
            ad_result = anderson(test_data)
            shapiro_stat, shapiro_p = shapiro(test_data[:5000])
        else:
            # Not enough data for normality tests
            logger.warning("Not enough samples for normality tests")
            ad_result = type('obj', (object,), {'statistic': 0.0, 'critical_values': [0, 0, 0, 0, 0]})()
            shapiro_stat, shapiro_p = 0.0, 1.0

        statistics = {
            "n_samples": int(n),
            "mean": format_number(mean),
            "std_dev": format_number(std_dev),
            "variance": format_number(variance),
            "rms": format_number(rms),
            "min": format_number(float(min_val)),
            "max": format_number(float(max_val)),
            "range": format_number(float(max_val - min_val)),
            "zero_pct": format_number(100 * n_zeros / n),
            "skewness": format_number(val_skew),
            "kurtosis": format_number(val_kurt),
            "normality": {
                "anderson_darling_stat": format_number(ad_result.statistic),
                "anderson_darling_critical_5pct": format_number(ad_result.critical_values[2]),
                "shapiro_wilk_stat": format_number(shapiro_stat),
                "shapiro_wilk_p": float(shapiro_p),
                "is_normal": bool(shapiro_p > 0.05)
            }
        }

        self.results.statistics = statistics
        self.cache.set(self.config.segy_file, "statistics", statistics)

        logger.info(f"Mean: {mean:.2e}, Std: {std_dev:.2e}")
        return statistics

    def quality_checks(self) -> Dict:
        """Perform data quality assessment"""
        logger.info("Running quality checks...")
        self.progress.update(step=30, phase="Quality Checks")

        if not self.config.run_quality_checks:
            return {}

        cached = self.cache.get(self.config.segy_file, "quality")
        if cached:
            self.results.quality = cached
            return cached

        sample_idx = self.stratified_sampling()

        dead_traces = []
        clipped_traces = []
        low_snr_traces = []
        snr_values = []
        dead_coords = []

        for idx in tqdm(sample_idx, desc="Quality checks"):
            trace = self.f.trace[int(idx)]
            header = self.f.header[int(idx)]

            il = header[segyio.TraceField.INLINE_3D]
            xl = header[segyio.TraceField.CROSSLINE_3D]

            # Dead trace check
            if np.all(trace == 0):
                dead_traces.append(int(idx))
                dead_coords.append([int(il), int(xl)])
                continue

            # Clipping check
            max_amp = np.max(np.abs(trace))
            if max_amp > 0:
                n_clipped = np.sum(np.abs(trace) >= 0.99 * max_amp)
                if n_clipped / len(trace) > 0.01:
                    clipped_traces.append(int(idx))

            # SNR check
            snr = calculate_snr_coherence(trace)
            snr_values.append(snr)
            if snr < 10 and snr > -999:
                low_snr_traces.append(int(idx))

        n_checked = len(sample_idx)
        snr_values = [s for s in snr_values if s > -999]

        quality = {
            "traces_checked": int(n_checked),
            "dead_traces": {
                "count": len(dead_traces),
                "pct": format_number(100 * len(dead_traces) / n_checked),
                "extrapolated_total": int(self.n_traces * len(dead_traces) / n_checked)
            },
            "clipped_traces": {
                "count": len(clipped_traces),
                "pct": format_number(100 * len(clipped_traces) / n_checked)
            },
            "low_snr_traces": {
                "count": len(low_snr_traces),
                "pct": format_number(100 * len(low_snr_traces) / n_checked),
                "threshold_db": 10
            },
            "snr_statistics": {
                "mean_db": format_number(float(np.mean(snr_values))) if snr_values else 0,
                "median_db": format_number(float(np.median(snr_values))) if snr_values else 0,
                "std_db": format_number(float(np.std(snr_values))) if snr_values else 0,
                "min_db": format_number(float(np.min(snr_values))) if snr_values else 0,
                "max_db": format_number(float(np.max(snr_values))) if snr_values else 0,
                "method": "coherence-based (autocorrelation)"
            },
            "dead_trace_coordinates": dead_coords[:100]  # First 100 for reference
        }

        # Add warnings
        if quality["dead_traces"]["pct"] > 10:
            self.results.warnings.append(f"High dead trace percentage: {quality['dead_traces']['pct']}%")

        self.results.quality = quality
        self.cache.set(self.config.segy_file, "quality", quality)

        logger.info(f"Dead: {quality['dead_traces']['pct']}%, SNR: {quality['snr_statistics']['mean_db']} dB")
        return quality

    def frequency_analysis(self) -> Dict:
        """Perform spectral analysis with sensitivity testing"""
        logger.info("Analyzing frequency content...")
        self.progress.update(step=45, phase="Frequency Analysis")

        if not self.config.run_frequency_analysis:
            return {}

        cached = self.cache.get(self.config.segy_file, "spectral")
        if cached:
            self.results.spectral = cached
            # Extract values for resolution analysis
            self._peak_freq = float(cached.get("peak_frequency_hz", 25))
            self._f_low_3db = float(cached.get("f_low_3db_hz", 10))
            self._f_high_3db = float(cached.get("f_high_3db_hz", 80))
            self._bandwidth_3db = float(cached.get("bandwidth_3db_hz", 70))
            return cached

        n_sample = min(self.config.frequency_sample_traces, self.n_traces)
        sample_idx = np.random.choice(self.n_traces, n_sample, replace=False)

        all_spectra = []
        dom_freqs = []

        # Time-variant analysis
        shallow_spectra = []
        deep_spectra = []

        # Check if GPU is available and enabled
        use_gpu = self.config.use_gpu and GPU_AVAILABLE
        if use_gpu:
            logger.info("Using GPU acceleration for FFT analysis")

        # Batch processing for GPU efficiency
        if use_gpu:
            # Load traces in batches for GPU processing
            batch_size = min(500, n_sample)
            for batch_start in tqdm(range(0, n_sample, batch_size), desc="GPU FFT analysis"):
                batch_end = min(batch_start + batch_size, n_sample)
                batch_idx = sample_idx[batch_start:batch_end]

                # Load batch of traces
                traces_batch = np.array([self.f.trace[int(idx)] for idx in batch_idx])
                traces_gpu = cp.asarray(traces_batch)

                # Batch FFT on GPU
                fft_batch = cp.fft.rfft(traces_gpu, axis=1)
                spectra_batch = cp.abs(fft_batch)
                spectra_cpu = cp.asnumpy(spectra_batch)

                freqs = np.fft.rfftfreq(traces_batch.shape[1], d=self.sample_rate/1000)

                for i, spectrum in enumerate(spectra_cpu):
                    all_spectra.append(spectrum)
                    if len(spectrum) > 1:
                        dom_idx = np.argmax(spectrum[1:]) + 1
                        dom_freqs.append(freqs[dom_idx])

                # Time-variant analysis on GPU
                n_third = traces_batch.shape[1] // 3
                shallow_batch = traces_gpu[:, :n_third]
                deep_batch = traces_gpu[:, 2*n_third:]

                shallow_fft = cp.abs(cp.fft.rfft(shallow_batch, axis=1))
                deep_fft = cp.abs(cp.fft.rfft(deep_batch, axis=1))

                shallow_spectra.extend(cp.asnumpy(shallow_fft))
                deep_spectra.extend(cp.asnumpy(deep_fft))

            cp.cuda.Stream.null.synchronize()
        else:
            # CPU fallback
            for idx in tqdm(sample_idx, desc="FFT analysis"):
                trace = self.f.trace[int(idx)]

                # Full trace FFT
                fft_result = np.fft.rfft(trace)
                freqs = np.fft.rfftfreq(len(trace), d=self.sample_rate/1000)
                spectrum = np.abs(fft_result)

                all_spectra.append(spectrum)

                if len(spectrum) > 1:
                    dom_idx = np.argmax(spectrum[1:]) + 1
                    dom_freqs.append(freqs[dom_idx])

                # Time-variant (thirds)
                n_third = len(trace) // 3
                shallow = trace[:n_third]
                deep = trace[2*n_third:]

                shallow_spectra.append(np.abs(np.fft.rfft(shallow)))
                deep_spectra.append(np.abs(np.fft.rfft(deep)))

        # Average spectrum
        min_len = min(len(s) for s in all_spectra)
        avg_spectrum = np.mean([s[:min_len] for s in all_spectra], axis=0)
        freqs = freqs[:min_len]

        # Peak frequency
        peak_idx = np.argmax(avg_spectrum[1:]) + 1
        peak_freq = freqs[peak_idx]

        # Bandwidth
        peak_amp = avg_spectrum[peak_idx]

        # -3dB
        threshold_3db = peak_amp / np.sqrt(2)
        above_3db = avg_spectrum >= threshold_3db
        freq_3db = freqs[above_3db]
        f_low_3db = freq_3db[0] if len(freq_3db) > 0 else 0
        f_high_3db = freq_3db[-1] if len(freq_3db) > 0 else 0
        bandwidth_3db = f_high_3db - f_low_3db

        # -6dB
        threshold_6db = peak_amp / 2
        above_6db = avg_spectrum >= threshold_6db
        freq_6db = freqs[above_6db]
        f_low_6db = freq_6db[0] if len(freq_6db) > 0 else 0
        f_high_6db = freq_6db[-1] if len(freq_6db) > 0 else 0
        bandwidth_6db = f_high_6db - f_low_6db

        # Time-variant decay
        min_len_tv = min(min(len(s) for s in shallow_spectra), min(len(s) for s in deep_spectra))
        avg_shallow = np.mean([s[:min_len_tv] for s in shallow_spectra], axis=0)
        avg_deep = np.mean([s[:min_len_tv] for s in deep_spectra], axis=0)

        freqs_tv = np.fft.rfftfreq(len(shallow_spectra[0])*2-1, d=self.sample_rate/1000)[:min_len_tv]

        peak_shallow = freqs_tv[np.argmax(avg_shallow[1:]) + 1] if len(avg_shallow) > 1 else 0
        peak_deep = freqs_tv[np.argmax(avg_deep[1:]) + 1] if len(avg_deep) > 1 else 0

        decay_pct = 100 * (peak_shallow - peak_deep) / peak_shallow if peak_shallow > 0 else 0

        spectral = {
            "peak_frequency_hz": format_number(float(peak_freq)),
            "bandwidth_3db_hz": format_number(float(bandwidth_3db)),
            "f_low_3db_hz": format_number(float(f_low_3db)),
            "f_high_3db_hz": format_number(float(f_high_3db)),
            "bandwidth_6db_hz": format_number(float(bandwidth_6db)),
            "f_low_6db_hz": format_number(float(f_low_6db)),
            "f_high_6db_hz": format_number(float(f_high_6db)),
            "dominant_frequency": {
                "mean_hz": format_number(float(np.mean(dom_freqs))),
                "std_hz": format_number(float(np.std(dom_freqs))),
                "min_hz": format_number(float(np.min(dom_freqs))),
                "max_hz": format_number(float(np.max(dom_freqs)))
            },
            "time_variant": {
                "shallow_peak_hz": format_number(float(peak_shallow)),
                "deep_peak_hz": format_number(float(peak_deep)),
                "decay_pct": format_number(float(decay_pct)),
                "anomalous": bool(decay_pct < -10),
                "note": "Negative decay may indicate spectral balancing in processing"
            },
            "nyquist_hz": format_number(float(freqs[-1])),
            "traces_analyzed": int(n_sample)
        }

        # Warnings
        if spectral["time_variant"]["anomalous"]:
            self.results.warnings.append("Spectral decay is anomalous - may indicate processing artifacts")

        self.results.spectral = spectral
        self.cache.set(self.config.segy_file, "spectral", spectral)

        # Store for resolution analysis
        self._peak_freq = peak_freq
        self._f_low_3db = f_low_3db
        self._f_high_3db = f_high_3db
        self._bandwidth_3db = bandwidth_3db

        logger.info(f"Peak: {peak_freq:.1f} Hz, Bandwidth: {bandwidth_3db:.1f} Hz")
        return spectral

    def resolution_analysis(self) -> Dict:
        """Calculate resolution with comparison to assumptions"""
        logger.info("Analyzing resolution...")
        self.progress.update(step=55, phase="Resolution Analysis")

        if not self.config.run_resolution_analysis:
            return {}

        # Need spectral results
        if not hasattr(self, '_peak_freq'):
            self.frequency_analysis()

        v = self.config.avg_velocity
        peak_freq = self._peak_freq
        f_low = self._f_low_3db
        f_high = self._f_high_3db

        # Vertical resolution
        wavelength_peak = v / peak_freq if peak_freq > 0 else 0
        res_peak = wavelength_peak / 4

        wavelength_best = v / f_high if f_high > 0 else 0
        res_best = wavelength_best / 4

        wavelength_worst = v / f_low if f_low > 0 else 0
        res_worst = wavelength_worst / 4

        # Compare to assumptions
        assumed_freq = 25  # Hz typical
        assumed_res = v / (4 * assumed_freq)
        improvement_factor = assumed_res / res_peak if res_peak > 0 else 0

        # Lateral resolution (Fresnel)
        depths = [1000, 2000, 3000, 5000]
        fresnel = []
        for z in depths:
            r_F = np.sqrt(v * z / peak_freq) if peak_freq > 0 else 0
            fresnel.append({
                "depth_m": z,
                "radius_m": format_number(r_F),
                "diameter_m": format_number(2 * r_F)
            })

        # Spatial aliasing
        bin_size = 25  # nominal
        nyquist_wavelength = 2 * bin_size
        nyquist_freq = v / nyquist_wavelength
        aliasing_risk = "HIGH" if f_high > nyquist_freq else "LOW"

        resolution = {
            "vertical": {
                "at_peak_freq_m": format_number(res_peak),
                "best_case_m": format_number(res_best),
                "worst_case_m": format_number(res_worst),
                "method": "lambda/4 criterion"
            },
            "lateral_fresnel": fresnel,
            "comparison_to_assumptions": {
                "assumed_frequency_hz": assumed_freq,
                "assumed_resolution_m": format_number(assumed_res),
                "measured_frequency_hz": format_number(float(peak_freq)),
                "measured_resolution_m": format_number(res_peak),
                "factor_coarser": format_number(improvement_factor)
            },
            "spatial_aliasing": {
                "bin_size_m": bin_size,
                "nyquist_freq_hz": format_number(nyquist_freq),
                "max_observed_freq_hz": format_number(float(f_high)),
                "risk": aliasing_risk
            },
            "velocity_used_ms": float(v)
        }

        self.results.resolution = resolution

        logger.info(f"Vertical resolution: {res_peak:.1f} m (peak), {res_best:.1f}-{res_worst:.1f} m (range)")
        return resolution

    def attribute_analysis(self) -> Dict:
        """Compute seismic attributes with correlations"""
        logger.info("Computing attributes...")
        self.progress.update(step=70, phase="Attribute Analysis")

        if not self.config.run_attribute_analysis:
            return {}

        cached = self.cache.get(self.config.segy_file, "attributes")
        if cached:
            self.results.attributes = cached
            return cached

        n_sample = min(self.config.attribute_sample_traces, self.n_traces)
        sample_idx = np.random.choice(self.n_traces, n_sample, replace=False)

        rms_amps = []
        inst_amps = []
        dom_freqs = []
        zcr_values = []
        inlines = []
        xlines = []

        for idx in tqdm(sample_idx, desc="Attributes"):
            trace = self.f.trace[int(idx)]
            header = self.f.header[int(idx)]

            inlines.append(header[segyio.TraceField.INLINE_3D])
            xlines.append(header[segyio.TraceField.CROSSLINE_3D])

            # RMS
            rms_amps.append(np.sqrt(np.mean(trace**2)))

            # Instantaneous amplitude
            analytic = signal.hilbert(trace)
            inst_amps.append(np.mean(np.abs(analytic)))

            # Dominant frequency
            fft_result = np.fft.rfft(trace)
            freqs = np.fft.rfftfreq(len(trace), d=self.sample_rate/1000)
            spectrum = np.abs(fft_result)
            if len(spectrum) > 1:
                dom_freqs.append(freqs[np.argmax(spectrum[1:]) + 1])
            else:
                dom_freqs.append(0)

            # Zero crossing rate
            zcr = np.sum(np.diff(np.sign(trace)) != 0) / len(trace)
            zcr_values.append(zcr)

        rms_amps = np.array(rms_amps)
        inst_amps = np.array(inst_amps)
        dom_freqs = np.array(dom_freqs)
        zcr_values = np.array(zcr_values)
        inlines = np.array(inlines)
        xlines = np.array(xlines)

        # Statistics
        def attr_stats(data, name):
            return {
                "mean": format_number(float(np.mean(data))),
                "std": format_number(float(np.std(data))),
                "cv_pct": format_number(100 * float(np.std(data) / np.mean(data))) if np.mean(data) != 0 else 0,
                "min": format_number(float(np.min(data))),
                "max": format_number(float(np.max(data)))
            }

        # Correlations with bootstrap CI
        def correlation_with_ci(x, y, name):
            r, p = pearsonr(x, y)

            # Bootstrap
            n_boot = 1000
            boot_r = []
            for _ in range(n_boot):
                idx = np.random.choice(len(x), len(x), replace=True)
                r_boot, _ = pearsonr(x[idx], y[idx])
                boot_r.append(r_boot)

            ci_low = np.percentile(boot_r, 2.5)
            ci_high = np.percentile(boot_r, 97.5)

            return {
                "pair": name,
                "r": format_number(float(r)),
                "r2": format_number(float(r**2)),
                "p_value": float(p),
                "ci_95": [format_number(float(ci_low)), format_number(float(ci_high))]
            }

        attributes = {
            "rms_amplitude": attr_stats(rms_amps, "RMS Amplitude"),
            "instantaneous_amplitude": attr_stats(inst_amps, "Instantaneous Amplitude"),
            "dominant_frequency": attr_stats(dom_freqs, "Dominant Frequency (Hz)"),
            "zero_crossing_rate": attr_stats(zcr_values, "Zero Crossing Rate"),
            "correlations": [
                correlation_with_ci(rms_amps, dom_freqs, "RMS vs Frequency"),
                correlation_with_ci(rms_amps, inst_amps, "RMS vs Instantaneous"),
                correlation_with_ci(dom_freqs, zcr_values, "Frequency vs ZCR")
            ],
            "traces_analyzed": int(n_sample)
        }

        # Store for spatial analysis
        self._attr_data = {
            'rms': rms_amps,
            'inst': inst_amps,
            'freq': dom_freqs,
            'zcr': zcr_values,
            'il': inlines,
            'xl': xlines
        }

        self.results.attributes = attributes
        self.cache.set(self.config.segy_file, "attributes", attributes)

        logger.info(f"RMS mean: {attributes['rms_amplitude']['mean']}")
        return attributes

    def spatial_analysis(self) -> Dict:
        """Analyze spatial patterns with quadrant analysis"""
        logger.info("Analyzing spatial patterns...")
        self.progress.update(step=85, phase="Spatial Analysis")

        if not self.config.run_spatial_analysis:
            return {}

        # Need attribute data
        if not hasattr(self, '_attr_data'):
            self.attribute_analysis()

        data = self._attr_data
        inlines = data['il']
        xlines = data['xl']
        rms = data['rms']
        freq = data['freq']

        # Quadrant analysis
        il_median = np.median(inlines)
        xl_median = np.median(xlines)

        quadrants = {
            'NW': (inlines >= il_median) & (xlines < xl_median),
            'NE': (inlines >= il_median) & (xlines >= xl_median),
            'SW': (inlines < il_median) & (xlines < xl_median),
            'SE': (inlines < il_median) & (xlines >= xl_median)
        }

        quad_stats = {}
        for name, mask in quadrants.items():
            quad_stats[name] = {
                "rms_mean": format_number(float(np.mean(rms[mask]))),
                "freq_mean": format_number(float(np.mean(freq[mask]))),
                "trace_count": int(np.sum(mask))
            }

        # ANOVA
        rms_groups = [rms[mask] for mask in quadrants.values()]
        freq_groups = [freq[mask] for mask in quadrants.values()]

        f_rms, p_rms = stats.f_oneway(*rms_groups)
        f_freq, p_freq = stats.f_oneway(*freq_groups)

        spatial = {
            "quadrant_statistics": quad_stats,
            "anova": {
                "rms": {
                    "f_statistic": format_number(float(f_rms)),
                    "p_value": float(p_rms),
                    "significant": bool(p_rms < 0.05)
                },
                "frequency": {
                    "f_statistic": format_number(float(f_freq)),
                    "p_value": float(p_freq),
                    "significant": bool(p_freq < 0.05)
                }
            },
            "interpretation": "Significant spatial variations detected" if p_rms < 0.05 or p_freq < 0.05 else "No significant spatial variations"
        }

        self.results.spatial = spatial

        logger.info(f"Spatial ANOVA - RMS p={p_rms:.3e}, Freq p={p_freq:.3e}")
        return spatial

    def generate_figures(self):
        """Generate all visualization figures"""
        if not self.config.save_figures:
            return

        logger.info("Generating figures...")
        self.progress.update(step=90, phase="Generating Figures")

        plt.style.use('seaborn-v0_8-whitegrid')

        figures_dir = self.output_dir / "figures"
        figures_dir.mkdir(exist_ok=True)

        # Store figure paths
        self.results.output_files["figures"] = []

        # Figure 1: Survey Overview
        if hasattr(self, '_attr_data'):
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))

            data = self._attr_data

            # RMS map
            sc1 = axes[0, 0].scatter(data['xl'], data['il'], c=data['rms'],
                                     cmap='viridis', s=1, alpha=0.5)
            axes[0, 0].set_xlabel('Crossline')
            axes[0, 0].set_ylabel('Inline')
            axes[0, 0].set_title('RMS Amplitude Map')
            plt.colorbar(sc1, ax=axes[0, 0])

            # Frequency map
            sc2 = axes[0, 1].scatter(data['xl'], data['il'], c=data['freq'],
                                     cmap='plasma', s=1, alpha=0.5)
            axes[0, 1].set_xlabel('Crossline')
            axes[0, 1].set_ylabel('Inline')
            axes[0, 1].set_title('Dominant Frequency Map')
            plt.colorbar(sc2, ax=axes[0, 1])

            # RMS histogram
            axes[1, 0].hist(data['rms'], bins=50, edgecolor='black', alpha=0.7)
            axes[1, 0].set_xlabel('RMS Amplitude')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('RMS Amplitude Distribution')

            # Frequency histogram
            axes[1, 1].hist(data['freq'], bins=50, edgecolor='black', alpha=0.7, color='orange')
            axes[1, 1].set_xlabel('Dominant Frequency (Hz)')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Dominant Frequency Distribution')

            plt.tight_layout()
            fig_path = figures_dir / f"survey_overview.{self.config.figure_format}"
            plt.savefig(fig_path, dpi=self.config.figure_dpi, bbox_inches='tight')
            plt.close()
            self.results.output_files["figures"].append(str(fig_path))

        # Figure 2: Quality Summary
        if self.results.quality:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            quality = self.results.quality

            # Quality issues bar chart
            issues = ['Dead', 'Clipped', 'Low SNR']
            pcts = [
                quality['dead_traces']['pct'],
                quality['clipped_traces']['pct'],
                quality['low_snr_traces']['pct']
            ]

            colors = ['red' if p > 10 else 'orange' if p > 5 else 'green' for p in pcts]
            axes[0].bar(issues, pcts, color=colors, edgecolor='black')
            axes[0].set_ylabel('Percentage (%)')
            axes[0].set_title('Data Quality Issues')
            axes[0].axhline(y=10, color='red', linestyle='--', label='Warning threshold')

            # Dead trace locations
            if quality.get('dead_trace_coordinates'):
                coords = np.array(quality['dead_trace_coordinates'])
                if len(coords) > 0:
                    axes[1].scatter(coords[:, 1], coords[:, 0], c='red', s=10, alpha=0.5)
                    axes[1].set_xlabel('Crossline')
                    axes[1].set_ylabel('Inline')
                    axes[1].set_title('Dead Trace Locations')

            plt.tight_layout()
            fig_path = figures_dir / f"quality_summary.{self.config.figure_format}"
            plt.savefig(fig_path, dpi=self.config.figure_dpi, bbox_inches='tight')
            plt.close()
            self.results.output_files["figures"].append(str(fig_path))

        logger.info(f"Saved {len(self.results.output_files.get('figures', []))} figures")

    def generate_recommendations(self):
        """Generate recommendations based on analysis"""
        logger.info("Generating recommendations...")

        recs = []

        # Quality-based
        if self.results.quality:
            q = self.results.quality
            if q['dead_traces']['pct'] > 10:
                recs.append(f"Address dead traces ({q['dead_traces']['pct']}%) - consider interpolation or reprocessing")
            if q['snr_statistics']['mean_db'] < 15:
                recs.append(f"Consider noise attenuation - mean SNR is {q['snr_statistics']['mean_db']} dB")

        # Resolution-based
        if self.results.resolution:
            r = self.results.resolution
            if r['vertical']['at_peak_freq_m'] > 30:
                recs.append(f"Account for {r['vertical']['at_peak_freq_m']}m resolution limit in interpretation")

        # Spectral-based
        if self.results.spectral:
            s = self.results.spectral
            if s['peak_frequency_hz'] < 15:
                recs.append(f"Low frequency content ({s['peak_frequency_hz']} Hz) - typical of deep targets")
            if s['time_variant']['anomalous']:
                recs.append("Investigate spectral artifacts in time-variant analysis")

        # Spatial-based
        if self.results.spatial:
            sp = self.results.spatial
            if sp['anova']['rms']['significant']:
                recs.append("Investigate spatial amplitude variations across survey")

        # Standard recommendations
        recs.append("Integrate well control for velocity/impedance calibration")
        recs.append("Consider multiple attributes for reservoir characterization")

        self.results.recommendations = recs

    # ========================================================================
    # MAIN PIPELINE
    # ========================================================================

    def run(self) -> EDAResults:
        """Run complete EDA pipeline"""
        start_time = time.time()

        logger.info("="*60)
        logger.info("SEISMIC EDA AUTOMATION v5.0")
        logger.info(f"Input: {self.config.segy_file}")
        logger.info("="*60)

        self.progress.update(step=0, phase="Starting")

        # Run analyses
        self.load_metadata()
        self.compute_statistics()
        self.quality_checks()
        self.frequency_analysis()
        self.resolution_analysis()
        self.attribute_analysis()
        self.spatial_analysis()

        # Generate outputs
        self.generate_figures()
        self.generate_recommendations()

        # Save results
        self.progress.update(step=95, phase="Saving Results")

        # Save JSON results
        json_path = self.output_dir / "eda_results.json"
        self.results.to_json(str(json_path))
        self.results.output_files["results_json"] = str(json_path)

        # Save config used
        config_path = self.output_dir / "config_used.json"
        self.config.save(str(config_path))
        self.results.output_files["config"] = str(config_path)

        # Processing time
        self.results.processing_time_seconds = round(time.time() - start_time, 2)

        self.progress.update(step=100, phase="Complete")

        logger.info("="*60)
        logger.info(f"EDA COMPLETE in {self.results.processing_time_seconds:.1f}s")
        logger.info(f"Results: {json_path}")
        logger.info("="*60)

        return self.results


# ============================================================================
# CLI INTERFACE
# ============================================================================

def create_default_config(output_path: str):
    """Create a default configuration file"""
    config = EDAConfig()
    config.save(output_path)
    print(f"Default config saved to: {output_path}")


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Seismic EDA Automation Framework v5.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with SEGY file directly
  python seismic_eda_automation.py input.segy -o results/

  # Run with config file
  python seismic_eda_automation.py -c config.json

  # Generate default config
  python seismic_eda_automation.py --create-config my_config.json

  # With webhook for n8n
  python seismic_eda_automation.py input.segy --webhook http://n8n:5678/webhook/eda
        """
    )

    parser.add_argument('segy_file', nargs='?', help='Path to SEGY file')
    parser.add_argument('-c', '--config', help='Path to JSON config file')
    parser.add_argument('-o', '--output', default='eda_outputs', help='Output directory')
    parser.add_argument('--output-dir', dest='output', help='Output directory (alias for -o)')
    parser.add_argument('--webhook', help='Webhook URL for progress callbacks')
    parser.add_argument('--create-config', metavar='PATH', help='Create default config file')
    parser.add_argument('--no-cache', action='store_true', help='Disable caching')
    parser.add_argument('--no-figures', action='store_true', help='Skip figure generation')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create default config
    if args.create_config:
        create_default_config(args.create_config)
        return

    # Load or create config
    if args.config:
        config = EDAConfig.from_json(args.config)
    elif args.segy_file:
        config = EDAConfig(segy_file=args.segy_file)
    else:
        parser.print_help()
        sys.exit(1)

    # Override with CLI args
    config.output_dir = args.output
    if args.webhook:
        config.webhook_url = args.webhook
    if args.no_cache:
        config.enable_cache = False
    if args.no_figures:
        config.save_figures = False

    # Run EDA
    with SeismicEDAAutomation(config) as eda:
        results = eda.run()

    # Print summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Processing time: {results.processing_time_seconds}s")
    print(f"Traces analyzed: {results.survey.get('total_traces', 'N/A'):,}")
    print(f"Survey area: {results.survey.get('survey_area_km2', 'N/A')} km²")
    print(f"Dead traces: {results.quality.get('dead_traces', {}).get('pct', 'N/A')}%")
    print(f"Peak frequency: {results.spectral.get('peak_frequency_hz', 'N/A')} Hz")
    print(f"Vertical resolution: {results.resolution.get('vertical', {}).get('at_peak_freq_m', 'N/A')} m")
    print(f"\nWarnings: {len(results.warnings)}")
    for w in results.warnings:
        print(f"  - {w}")
    print(f"\nResults saved to: {config.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
