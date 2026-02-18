"""
Dead Trace Detection and Removal Automation Framework
Version 5.0 - Production-Ready for Automated Workflows

Features:
- Unified detection + removal pipeline
- Configuration-driven (JSON/CLI)
- Structured JSON output for n8n/automation
- REST API endpoint support
- Webhook progress callbacks
- Multiple interpolation methods
- Caching for efficiency
- Proper logging

Methodology:
- Multi-criteria adaptive detection (Yilmaz, 2001)
- DBSCAN spatial clustering (Ester et al., 1996)
- Interpolation comparison framework

Author: Moses Ekene Obasi
PhD Research - University of Calabar
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, signal, ndimage
from scipy.interpolate import griddata
from sklearn.cluster import DBSCAN
from collections import Counter
import warnings
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Any, Union, Tuple
from enum import Enum
import json
import logging
import hashlib
import pickle
import time
import argparse
import sys
from datetime import datetime

# Conditional imports
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
    print("GPU acceleration enabled for dead trace processing")
except ImportError:
    GPU_AVAILABLE = False
    cp = None

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('DeadTrace')


# ============================================================================
# CONFIGURATION
# ============================================================================

class InterpolationMethod(Enum):
    """Available interpolation methods"""
    NONE = "none"  # Just remove, don't interpolate
    NEAREST = "nearest"
    LINEAR = "linear"
    CUBIC = "cubic"
    MEDIAN = "median"
    KRIGING = "kriging"  # Future: Gaussian process


@dataclass
class DeadTraceConfig:
    """Configuration for dead trace detection and removal"""
    # Input/Output
    input_segy: str = ""
    output_segy: str = ""  # If empty, generates based on input
    output_dir: str = "dead_trace_outputs"

    # Detection thresholds (None = adaptive)
    threshold_variance: Optional[float] = None
    threshold_rms: Optional[float] = None
    threshold_zeros: float = 0.95  # >95% zeros = dead
    use_adaptive_thresholds: bool = True
    adaptive_sigma: float = 3.0  # N sigma below mean

    # Clustering
    dbscan_eps: float = 10.0  # IL/XL space
    dbscan_min_samples: int = 5

    # Interpolation
    interpolation_method: str = "linear"
    interpolate_isolated: bool = True  # Interpolate isolated, remove clustered
    max_cluster_size_for_interpolation: int = 20  # Clusters larger = remove

    # Processing
    sample_size_for_thresholds: int = 10000
    chunk_size: int = 10000

    # Output options
    save_figures: bool = True
    figure_dpi: int = 300
    generate_cleaned_segy: bool = True

    # Automation
    webhook_url: Optional[str] = None
    webhook_auth: Optional[str] = None
    progress_interval: int = 10

    # Caching
    enable_cache: bool = True
    cache_dir: str = ".dead_trace_cache"

    @classmethod
    def from_json(cls, path: str) -> 'DeadTraceConfig':
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_dict(cls, data: Dict) -> 'DeadTraceConfig':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> Dict:
        return asdict(self)

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class DeadTraceResults:
    """Structured results container"""
    # Metadata
    timestamp: str = ""
    input_file: str = ""
    output_file: str = ""
    processing_time_seconds: float = 0.0
    version: str = "5.0"

    # Survey info
    survey: Dict = field(default_factory=dict)

    # Detection results
    detection: Dict = field(default_factory=dict)

    # Clustering results
    clustering: Dict = field(default_factory=dict)

    # Thresholds used
    thresholds: Dict = field(default_factory=dict)

    # Removal results
    removal: Dict = field(default_factory=dict)

    # Interpolation results
    interpolation: Dict = field(default_factory=dict)

    # Quality metrics
    quality: Dict = field(default_factory=dict)

    # Recommendations
    recommendations: List[str] = field(default_factory=list)

    # Warnings
    warnings: List[str] = field(default_factory=list)

    # Output files
    output_files: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)

    def to_json(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def from_json(cls, path: str) -> 'DeadTraceResults':
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(**data)


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


class ProgressTracker:
    """Track and report progress"""

    def __init__(self, config: DeadTraceConfig, total_steps: int = 100):
        self.config = config
        self.total_steps = total_steps
        self.current_step = 0
        self.current_phase = ""
        self.start_time = time.time()
        self._last_reported = 0

    def update(self, step: int = None, phase: str = None, message: str = None):
        if step is not None:
            self.current_step = step
        if phase is not None:
            self.current_phase = phase

        progress_pct = (self.current_step / self.total_steps) * 100

        if progress_pct - self._last_reported >= self.config.progress_interval:
            self._last_reported = progress_pct
            self._report_progress(progress_pct, message)

    def _report_progress(self, progress_pct: float, message: str = None):
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
                requests.post(self.config.webhook_url, json=payload, headers=headers, timeout=5)
            except Exception as e:
                logger.warning(f"Webhook failed: {e}")


class ResultCache:
    """Cache intermediate results"""

    def __init__(self, config: DeadTraceConfig):
        self.config = config
        self.cache_dir = Path(config.cache_dir)
        if config.enable_cache:
            self.cache_dir.mkdir(exist_ok=True)

    def _get_key(self, segy_file: str, operation: str) -> str:
        file_path = Path(segy_file)
        if file_path.exists():
            stat = file_path.stat()
            hash_source = f"{segy_file}:{stat.st_size}:{stat.st_mtime}:{operation}"
        else:
            hash_source = f"{segy_file}:{operation}"
        return hashlib.md5(hash_source.encode()).hexdigest()

    def get(self, segy_file: str, operation: str) -> Optional[Any]:
        if not self.config.enable_cache:
            return None
        key = self._get_key(segy_file, operation)
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except:
                return None
        return None

    def set(self, segy_file: str, operation: str, data: Any):
        if not self.config.enable_cache:
            return
        key = self._get_key(segy_file, operation)
        cache_file = self.cache_dir / f"{key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.warning(f"Cache write failed: {e}")


# ============================================================================
# MAIN CLASS
# ============================================================================

class DeadTraceAutomation:
    """
    Production-ready Dead Trace Detection and Removal

    Combines detection, clustering, and removal into a unified pipeline
    with full automation support.
    """

    def __init__(self, config: Union[DeadTraceConfig, Dict, str]):
        """
        Initialize dead trace processor

        Parameters:
        -----------
        config : DeadTraceConfig, dict, or str (path to JSON or SEGY)
        """
        # Load configuration
        if isinstance(config, str):
            if config.endswith('.json'):
                self.config = DeadTraceConfig.from_json(config)
            else:
                # Assume SEGY path
                self.config = DeadTraceConfig(input_segy=config)
        elif isinstance(config, dict):
            self.config = DeadTraceConfig.from_dict(config)
        else:
            self.config = config

        if not SEGYIO_AVAILABLE:
            raise ImportError("segyio is required")

        if not self.config.input_segy:
            raise ValueError("input_segy must be specified")

        # Generate output SEGY path if not specified - write to output_dir, not input folder
        if not self.config.output_segy:
            input_path = Path(self.config.input_segy)
            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            suffix = input_path.suffix if input_path.suffix else ".segy"
            self.config.output_segy = str(output_dir / f"{input_path.stem}_cleaned{suffix}")

        # Setup
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.cache = ResultCache(self.config)
        self.progress = ProgressTracker(self.config)
        self.results = DeadTraceResults(
            timestamp=datetime.now().isoformat(),
            input_file=self.config.input_segy,
            output_file=self.config.output_segy
        )

        self.f = None

        # Data storage
        self.n_traces = 0
        self.n_samples = 0
        self.sample_rate = 0.0
        self.dead_indices = []
        self.thresholds = {}

    def __enter__(self):
        self.f = segyio.open(self.config.input_segy, ignore_geometry=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.f:
            self.f.close()

    # ========================================================================
    # DETECTION METHODS
    # ========================================================================

    def load_metadata(self) -> Dict:
        """Load survey metadata"""
        logger.info("Loading metadata...")
        self.progress.update(step=5, phase="Loading Metadata")

        self.n_traces = self.f.tracecount
        self.n_samples = self.f.samples.size
        self.sample_rate = segyio.tools.dt(self.f) / 1000

        # Sample for geometry
        sample_size = min(10000, self.n_traces)
        sample_idx = np.linspace(0, self.n_traces - 1, sample_size, dtype=int)

        inlines, xlines = [], []
        for idx in sample_idx:
            header = self.f.header[int(idx)]
            inlines.append(header[segyio.TraceField.INLINE_3D])
            xlines.append(header[segyio.TraceField.CROSSLINE_3D])

        self.inline_range = (min(inlines), max(inlines))
        self.xline_range = (min(xlines), max(xlines))
        self.n_inlines = len(np.unique(inlines))
        self.n_xlines = len(np.unique(xlines))

        survey = {
            "n_traces": int(self.n_traces),
            "n_samples": int(self.n_samples),
            "sample_rate_ms": float(self.sample_rate),
            "inline_range": list(self.inline_range),
            "xline_range": list(self.xline_range),
            "n_inlines": int(self.n_inlines),
            "n_xlines": int(self.n_xlines),
            "expected_traces": int(self.n_inlines * self.n_xlines),
            "completeness_pct": format_number(100 * self.n_traces / (self.n_inlines * self.n_xlines))
        }

        self.results.survey = survey
        logger.info(f"Survey: {self.n_traces:,} traces, {survey['completeness_pct']}% complete")
        return survey

    def compute_adaptive_thresholds(self) -> Dict:
        """Compute adaptive thresholds from live trace statistics"""
        logger.info("Computing adaptive thresholds...")
        self.progress.update(step=15, phase="Computing Thresholds")

        if not self.config.use_adaptive_thresholds:
            self.thresholds = {
                'variance': self.config.threshold_variance or 1e-10,
                'rms': self.config.threshold_rms or 1e-8,
                'zeros': self.config.threshold_zeros,
                'method': 'manual'
            }
            self.results.thresholds = self.thresholds
            return self.thresholds

        # Check cache
        cached = self.cache.get(self.config.input_segy, "thresholds")
        if cached:
            self.thresholds = cached
            self.results.thresholds = cached
            return cached

        # Sample traces
        sample_size = min(self.config.sample_size_for_thresholds, self.n_traces)
        sample_idx = np.random.choice(self.n_traces, sample_size, replace=False)

        live_variance = []
        live_rms = []

        for idx in tqdm(sample_idx, desc="Sampling for thresholds"):
            trace = self.f.trace[int(idx)]

            # Skip obvious dead traces
            if np.all(trace == 0) or np.sum(trace == 0) / len(trace) > 0.95:
                continue

            live_variance.append(np.var(trace))
            live_rms.append(np.sqrt(np.mean(trace ** 2)))

        if len(live_variance) < 100:
            logger.warning("Few live traces found for threshold computation")
            self.thresholds = {
                'variance': 1e-10,
                'rms': 1e-8,
                'zeros': self.config.threshold_zeros,
                'method': 'default (insufficient samples)'
            }
        else:
            sigma = self.config.adaptive_sigma
            var_mean = np.mean(live_variance)
            var_std = np.std(live_variance)
            rms_mean = np.mean(live_rms)
            rms_std = np.std(live_rms)

            self.thresholds = {
                'variance': max(var_mean - sigma * var_std, 1e-10),
                'rms': max(rms_mean - sigma * rms_std, 1e-8),
                'zeros': self.config.threshold_zeros,
                'method': f'adaptive (μ - {sigma}σ)',
                'live_variance_mean': format_number(var_mean),
                'live_variance_std': format_number(var_std),
                'live_rms_mean': format_number(rms_mean),
                'live_rms_std': format_number(rms_std),
                'n_live_samples': len(live_variance)
            }

        self.cache.set(self.config.input_segy, "thresholds", self.thresholds)
        self.results.thresholds = self.thresholds

        logger.info(f"Thresholds: variance < {self.thresholds['variance']:.2e}, RMS < {self.thresholds['rms']:.2e}")
        return self.thresholds

    def detect_dead_traces(self) -> Dict:
        """Multi-criteria dead trace detection"""
        logger.info("Detecting dead traces...")
        self.progress.update(step=30, phase="Detecting Dead Traces")

        # Check cache
        cached = self.cache.get(self.config.input_segy, "detection")
        if cached:
            self.dead_indices = cached['dead_indices']
            self.dead_by_criterion = cached['by_criterion']
            self.dead_coords = cached.get('coords', [])
            self.results.detection = cached
            return cached

        var_threshold = self.thresholds['variance']
        rms_threshold = self.thresholds['rms']
        zero_threshold = self.thresholds['zeros']

        # Initialize
        dead_indices = []
        dead_by_criterion = {
            'all_zero': [],
            'high_zeros': [],
            'low_variance': [],
            'low_rms': []
        }
        dead_coords = []

        for i in tqdm(range(self.n_traces), desc="Scanning traces"):
            trace = self.f.trace[i]
            is_dead = False

            # Criterion 1: All zeros
            if np.all(trace == 0):
                dead_by_criterion['all_zero'].append(i)
                is_dead = True
                dead_indices.append(i)

                # Get coordinates
                header = self.f.header[i]
                il = header[segyio.TraceField.INLINE_3D]
                xl = header[segyio.TraceField.CROSSLINE_3D]
                dead_coords.append([int(il), int(xl)])
                continue

            # Compute statistics
            variance = np.var(trace)
            zero_frac = np.sum(trace == 0) / len(trace)
            rms = np.sqrt(np.mean(trace ** 2))

            # Criterion 2: High zero content
            if zero_frac > zero_threshold:
                dead_by_criterion['high_zeros'].append(i)
                is_dead = True

            # Criterion 3: Low variance
            if variance < var_threshold:
                dead_by_criterion['low_variance'].append(i)
                is_dead = True

            # Criterion 4: Low RMS
            if rms < rms_threshold:
                dead_by_criterion['low_rms'].append(i)
                is_dead = True

            if is_dead and i not in dead_indices:
                dead_indices.append(i)
                header = self.f.header[i]
                il = header[segyio.TraceField.INLINE_3D]
                xl = header[segyio.TraceField.CROSSLINE_3D]
                dead_coords.append([int(il), int(xl)])

        # Sort and deduplicate
        dead_indices = sorted(list(set(dead_indices)))

        self.dead_indices = dead_indices
        self.dead_by_criterion = dead_by_criterion
        self.dead_coords = dead_coords

        n_dead = len(dead_indices)
        detection = {
            'n_total': int(self.n_traces),
            'n_dead': int(n_dead),
            'n_live': int(self.n_traces - n_dead),
            'pct_dead': format_number(100 * n_dead / self.n_traces),
            'pct_live': format_number(100 * (self.n_traces - n_dead) / self.n_traces),
            'by_criterion': {k: len(v) for k, v in dead_by_criterion.items()},
            'by_criterion_pct': {k: format_number(100 * len(v) / self.n_traces) for k, v in dead_by_criterion.items()},
            'dead_indices': dead_indices[:1000],  # First 1000 for reference
            'coords': dead_coords[:1000]  # First 1000 coordinates
        }

        self.cache.set(self.config.input_segy, "detection", detection)
        self.results.detection = detection

        logger.info(f"Detected {n_dead:,} dead traces ({detection['pct_dead']}%)")
        return detection

    def spatial_clustering(self) -> Dict:
        """DBSCAN clustering of dead traces"""
        logger.info("Analyzing spatial clustering...")
        self.progress.update(step=50, phase="Spatial Clustering")

        if len(self.dead_coords) < 10:
            clustering = {
                'n_clusters': 0,
                'n_isolated': len(self.dead_indices),
                'n_clustered': 0,
                'method': 'DBSCAN',
                'note': 'Too few dead traces for clustering'
            }
            self.results.clustering = clustering
            return clustering

        # Limit for memory
        coords = np.array(self.dead_coords[:50000])

        # DBSCAN
        clustering_model = DBSCAN(
            eps=self.config.dbscan_eps,
            min_samples=self.config.dbscan_min_samples
        ).fit(coords)

        labels = clustering_model.labels_

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_isolated = list(labels).count(-1)
        n_clustered = len(labels) - n_isolated

        # Cluster sizes
        cluster_sizes = Counter([l for l in labels if l != -1])
        largest_cluster = max(cluster_sizes.values()) if cluster_sizes else 0

        self.clustering_labels = labels
        self.clustering_coords = coords

        clustering = {
            'n_clusters': int(n_clusters),
            'n_isolated': int(n_isolated),
            'n_clustered': int(n_clustered),
            'pct_isolated': format_number(100 * n_isolated / len(labels)) if len(labels) > 0 else 0,
            'pct_clustered': format_number(100 * n_clustered / len(labels)) if len(labels) > 0 else 0,
            'largest_cluster_size': int(largest_cluster),
            'cluster_sizes': dict(Counter([int(s) for s in cluster_sizes.values()])),
            'method': 'DBSCAN',
            'eps': self.config.dbscan_eps,
            'min_samples': self.config.dbscan_min_samples
        }

        self.results.clustering = clustering

        logger.info(f"Found {n_clusters} clusters, {n_isolated} isolated ({clustering['pct_isolated']}%)")
        return clustering

    def compare_interpolation_methods(self) -> Dict:
        """Compare interpolation methods on sample dead traces"""
        logger.info("Comparing interpolation methods...")
        self.progress.update(step=60, phase="Interpolation Comparison")

        if not self.dead_indices:
            return {}

        # Select a sample dead trace with neighbors
        sample_idx = None
        for idx in self.dead_indices[len(self.dead_indices) // 2:]:
            if 10 < idx < self.n_traces - 10:
                sample_idx = idx
                break

        if sample_idx is None:
            return {'note': 'Could not find suitable sample trace'}

        # Get neighbors
        dead_set = set(self.dead_indices)
        neighbors = []
        neighbor_dists = []

        for offset in range(-10, 11):
            if offset == 0:
                continue
            idx = sample_idx + offset
            if 0 <= idx < self.n_traces and idx not in dead_set:
                neighbors.append(self.f.trace[idx])
                neighbor_dists.append(abs(offset))

        if len(neighbors) < 2:
            return {'note': 'Insufficient neighbors for comparison'}

        neighbors = np.array(neighbors)
        neighbor_dists = np.array(neighbor_dists)

        # Ground truth proxy
        ground_truth = np.mean(neighbors, axis=0)

        # Methods
        methods = {}

        # Nearest
        nearest = neighbors[np.argmin(neighbor_dists)]
        methods['nearest'] = nearest

        # Linear (distance-weighted)
        weights = 1.0 / (neighbor_dists + 1e-10)
        weights = weights / np.sum(weights)
        linear = np.sum(neighbors.T * weights, axis=1)
        methods['linear'] = linear

        # Median
        median = np.median(neighbors, axis=0)
        methods['median'] = median

        # Average of closest 3
        closest_idx = np.argsort(neighbor_dists)[:3]
        avg_close = np.mean(neighbors[closest_idx], axis=0)
        methods['avg_closest_3'] = avg_close

        # Compute metrics
        results = {}
        for name, interpolated in methods.items():
            corr = np.corrcoef(ground_truth, interpolated)[0, 1]
            rmse = np.sqrt(np.mean((ground_truth - interpolated) ** 2))

            # Spectral similarity
            gt_fft = np.abs(np.fft.rfft(ground_truth))
            int_fft = np.abs(np.fft.rfft(interpolated))
            spectral_corr = np.corrcoef(gt_fft, int_fft)[0, 1]

            results[name] = {
                'correlation': format_number(float(corr)),
                'rmse': format_number(float(rmse)),
                'spectral_correlation': format_number(float(spectral_corr))
            }

        # Find best
        best_method = max(results.items(), key=lambda x: x[1]['correlation'])

        interpolation = {
            'methods': results,
            'recommended': best_method[0],
            'recommended_correlation': best_method[1]['correlation'],
            'sample_trace_index': int(sample_idx),
            'n_neighbors_used': len(neighbors)
        }

        self.results.interpolation = interpolation

        logger.info(f"Best method: {best_method[0]} (r={best_method[1]['correlation']})")
        return interpolation

    def remove_dead_traces(self) -> Dict:
        """Remove dead traces and create cleaned SEGY"""
        if not self.config.generate_cleaned_segy:
            return {'note': 'SEGY generation disabled'}

        logger.info("Creating cleaned SEGY file...")
        self.progress.update(step=75, phase="Removing Dead Traces")

        dead_set = set(self.dead_indices)
        n_live = self.n_traces - len(self.dead_indices)

        # Create output specification
        spec = segyio.spec()
        spec.samples = self.f.samples
        spec.format = self.f.format
        spec.tracecount = n_live

        logger.info(f"Writing {n_live:,} live traces...")

        with segyio.create(self.config.output_segy, spec) as dst:
            # Copy headers
            dst.text[0] = self.f.text[0]
            dst.bin = self.f.bin
            dst.bin[segyio.BinField.Traces] = n_live

            # Write live traces
            write_idx = 0
            for read_idx in tqdm(range(self.n_traces), desc="Writing"):
                if read_idx not in dead_set:
                    dst.trace[write_idx] = self.f.trace[read_idx]
                    dst.header[write_idx] = self.f.header[read_idx]
                    write_idx += 1

        # Verify
        with segyio.open(self.config.output_segy, ignore_geometry=True) as verify:
            verified_count = verify.tracecount
            verification_passed = verified_count == n_live

        removal = {
            'input_file': self.config.input_segy,
            'output_file': self.config.output_segy,
            'original_traces': int(self.n_traces),
            'dead_traces_removed': int(len(self.dead_indices)),
            'live_traces_written': int(n_live),
            'verification_passed': verification_passed,
            'verified_trace_count': int(verified_count)
        }

        self.results.removal = removal
        self.results.output_files['cleaned_segy'] = self.config.output_segy

        logger.info(f"Created: {self.config.output_segy} ({n_live:,} traces)")
        return removal

    def compute_quality_metrics(self) -> Dict:
        """Compute quality metrics"""
        logger.info("Computing quality metrics...")
        self.progress.update(step=85, phase="Quality Metrics")

        expected_traces = self.n_inlines * self.n_xlines
        n_live = self.n_traces - len(self.dead_indices)

        original_completeness = 100 * self.n_traces / expected_traces
        effective_completeness = 100 * n_live / expected_traces

        quality = {
            'original_completeness_pct': format_number(original_completeness),
            'effective_completeness_pct': format_number(effective_completeness),
            'completeness_change_pct': format_number(effective_completeness - original_completeness),
            'data_recovery_pct': format_number(100 * n_live / self.n_traces),
            'expected_traces': int(expected_traces)
        }

        self.results.quality = quality

        logger.info(f"Effective completeness: {quality['effective_completeness_pct']}%")
        return quality

    def generate_figures(self):
        """Generate visualization figures"""
        if not self.config.save_figures:
            return

        logger.info("Generating figures...")
        self.progress.update(step=90, phase="Generating Figures")

        figures_dir = self.output_dir / "figures"
        figures_dir.mkdir(exist_ok=True)

        plt.style.use('seaborn-v0_8-whitegrid')

        # Figure 1: Detection Overview
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))

        # (a) Trace distribution
        ax = axes[0, 0]
        if self.dead_indices:
            ax.hist(self.dead_indices, bins=100, edgecolor='black', alpha=0.7, color='#e74c3c')
        ax.set_xlabel('Trace Index')
        ax.set_ylabel('Count')
        ax.set_title('(a) Dead Trace Distribution')
        ax.ticklabel_format(style='scientific', axis='x', scilimits=(0, 0))

        # (b) Criterion breakdown
        ax = axes[0, 1]
        criteria = list(self.dead_by_criterion.keys())
        counts = [len(self.dead_by_criterion[k]) for k in criteria]
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
        ax.barh(criteria, counts, color=colors, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Number of Traces')
        ax.set_title('(b) Detection by Criterion')

        # (c) Trace counts
        ax = axes[0, 2]
        categories = ['Total', 'Dead', 'Live']
        n_dead = len(self.dead_indices)
        n_live = self.n_traces - n_dead
        values = [self.n_traces, n_dead, n_live]
        colors_bar = ['#95a5a6', '#e74c3c', '#2ecc71']
        ax.bar(categories, values, color=colors_bar, alpha=0.7, edgecolor='black')
        ax.set_ylabel('Trace Count')
        ax.set_title('(c) Trace Count Summary')
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

        # (d) Spatial distribution
        ax = axes[1, 0]
        if len(self.dead_coords) > 0:
            coords = np.array(self.dead_coords[:10000])
            ax.scatter(coords[:, 1], coords[:, 0], c='red', s=1, alpha=0.5)
        ax.set_xlabel('Crossline')
        ax.set_ylabel('Inline')
        ax.set_title('(d) Dead Trace Locations')

        # (e) Clustering
        ax = axes[1, 1]
        if hasattr(self, 'clustering_labels') and hasattr(self, 'clustering_coords'):
            labels = self.clustering_labels
            coords = self.clustering_coords
            isolated = labels == -1
            clustered = labels != -1
            if np.any(isolated):
                ax.scatter(coords[isolated, 1], coords[isolated, 0], c='blue', s=5, alpha=0.5, label='Isolated')
            if np.any(clustered):
                ax.scatter(coords[clustered, 1], coords[clustered, 0], c='red', s=10, alpha=0.7, label='Clustered')
            ax.legend()
        ax.set_xlabel('Crossline')
        ax.set_ylabel('Inline')
        ax.set_title('(e) Spatial Clustering')

        # (f) Summary text
        ax = axes[1, 2]
        ax.axis('off')
        summary = f"""
DETECTION SUMMARY
{'─' * 40}

Total traces: {self.n_traces:,}
Dead traces: {n_dead:,} ({100 * n_dead / self.n_traces:.2f}%)
Live traces: {n_live:,} ({100 * n_live / self.n_traces:.2f}%)

Thresholds ({self.thresholds.get('method', 'N/A')}):
  Variance: < {self.thresholds.get('variance', 'N/A'):.2e}
  RMS: < {self.thresholds.get('rms', 'N/A'):.2e}
  Zero content: > {self.thresholds.get('zeros', 0.95):.1%}

Clustering:
  Clusters: {self.results.clustering.get('n_clusters', 0)}
  Isolated: {self.results.clustering.get('n_isolated', 0)}
  Clustered: {self.results.clustering.get('n_clustered', 0)}
"""
        ax.text(0.05, 0.95, summary, fontsize=9, family='monospace', verticalalignment='top')

        plt.tight_layout()
        fig_path = figures_dir / "detection_overview.png"
        plt.savefig(fig_path, dpi=self.config.figure_dpi, bbox_inches='tight')
        plt.close()

        self.results.output_files['figures'] = [str(fig_path)]
        logger.info(f"Saved: {fig_path}")

    def generate_recommendations(self):
        """Generate processing recommendations"""
        recs = []

        detection = self.results.detection
        clustering = self.results.clustering

        if detection.get('pct_dead', 0) > 15:
            recs.append(f"High dead trace percentage ({detection['pct_dead']}%). Consider reprocessing or investigating acquisition issues.")

        if clustering.get('pct_clustered', 0) > 80:
            recs.append("Most dead traces are clustered. Consider removing entire dead zones rather than interpolating.")

        if clustering.get('pct_isolated', 0) > 50:
            recs.append(f"Significant isolated dead traces ({clustering['pct_isolated']}%). Interpolation recommended.")

        if detection.get('pct_dead', 0) < 5:
            recs.append("Low dead trace percentage. Data quality is acceptable.")

        recs.append("Proceed to noise attenuation after dead trace removal.")
        recs.append("Validate output file loads correctly in interpretation software.")

        self.results.recommendations = recs

    # ========================================================================
    # MAIN PIPELINE
    # ========================================================================

    def run(self) -> DeadTraceResults:
        """Run complete dead trace detection and removal pipeline"""
        start_time = time.time()

        logger.info("=" * 60)
        logger.info("DEAD TRACE AUTOMATION v5.0")
        logger.info(f"Input: {self.config.input_segy}")
        logger.info("=" * 60)

        self.progress.update(step=0, phase="Starting")

        # Pipeline
        self.load_metadata()
        self.compute_adaptive_thresholds()
        self.detect_dead_traces()
        self.spatial_clustering()
        self.compare_interpolation_methods()
        self.remove_dead_traces()
        self.compute_quality_metrics()
        self.generate_figures()
        self.generate_recommendations()

        # Save results
        self.progress.update(step=95, phase="Saving Results")

        json_path = self.output_dir / "dead_trace_results.json"
        self.results.to_json(str(json_path))
        self.results.output_files['results_json'] = str(json_path)

        config_path = self.output_dir / "config_used.json"
        self.config.save(str(config_path))
        self.results.output_files['config'] = str(config_path)

        self.results.processing_time_seconds = round(time.time() - start_time, 2)

        self.progress.update(step=100, phase="Complete")

        logger.info("=" * 60)
        logger.info(f"COMPLETE in {self.results.processing_time_seconds:.1f}s")
        logger.info(f"Dead traces: {len(self.dead_indices):,} ({self.results.detection['pct_dead']}%)")
        logger.info(f"Output: {self.config.output_segy}")
        logger.info("=" * 60)

        return self.results


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Dead Trace Detection and Removal Automation v5.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python dead_trace_automation.py input.segy -o cleaned.segy
  python dead_trace_automation.py -c config.json
  python dead_trace_automation.py input.segy --webhook http://n8n/webhook
        """
    )

    parser.add_argument('input_segy', nargs='?', help='Path to input SEGY file')
    parser.add_argument('-o', '--output', help='Path to output cleaned SEGY')
    parser.add_argument('-c', '--config', help='Path to JSON config file')
    parser.add_argument('--output-dir', default='dead_trace_outputs', help='Output directory')
    parser.add_argument('--webhook', help='Webhook URL for progress callbacks')
    parser.add_argument('--no-segy', action='store_true', help='Skip SEGY generation')
    parser.add_argument('--no-figures', action='store_true', help='Skip figure generation')
    parser.add_argument('--no-cache', action='store_true', help='Disable caching')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load config
    if args.config:
        config = DeadTraceConfig.from_json(args.config)
    elif args.input_segy:
        config = DeadTraceConfig(input_segy=args.input_segy)
    else:
        parser.print_help()
        sys.exit(1)

    # Override with CLI args
    if args.output:
        config.output_segy = args.output
    config.output_dir = args.output_dir
    if args.webhook:
        config.webhook_url = args.webhook
    if args.no_segy:
        config.generate_cleaned_segy = False
    if args.no_figures:
        config.save_figures = False
    if args.no_cache:
        config.enable_cache = False

    # Run
    with DeadTraceAutomation(config) as processor:
        results = processor.run()

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Processing time: {results.processing_time_seconds}s")
    print(f"Total traces: {results.detection.get('n_total', 'N/A'):,}")
    print(f"Dead traces: {results.detection.get('n_dead', 'N/A'):,} ({results.detection.get('pct_dead', 'N/A')}%)")
    print(f"Live traces: {results.detection.get('n_live', 'N/A'):,}")
    print(f"Clusters: {results.clustering.get('n_clusters', 0)}")
    print(f"Isolated: {results.clustering.get('n_isolated', 0)}")
    print(f"\nRecommendations:")
    for rec in results.recommendations:
        print(f"  - {rec}")
    print(f"\nResults saved to: {config.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
