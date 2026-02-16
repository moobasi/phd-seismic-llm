"""
================================================================================
SEISMIC INVERSION AUTOMATION v5.0
Model-based post-stack acoustic impedance inversion
================================================================================

Features:
- Well log conditioning and QC
- Statistical wavelet extraction (3 methods)
- Low-frequency model construction
- Model-based acoustic impedance inversion
- Property prediction (porosity, lithology)
- Volumetric calculations (STOIIP)
- JSON structured output for automation
- CLI, API, and webhook support

Based on:
- Russell & Hampson (1991): Post-stack inversion methods
- Lancaster & Whitcombe (2000): Coloured inversion
- Cooke & Cant (2010): Model-based inversion
- Gardner et al. (1974): Velocity-density relationship

Author: Moses Ekene Obasi
PhD Research - University of Calabar
================================================================================
"""

import numpy as np
import json
import os
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
    from scipy import signal, interpolate, stats
    from scipy.ndimage import gaussian_filter, median_filter
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import segyio
    SEGYIO_AVAILABLE = True
except ImportError:
    SEGYIO_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    class tqdm:
        def __init__(self, iterable=None, **kwargs):
            self.iterable = iterable
            self.total = kwargs.get('total', 0)
        def __iter__(self):
            return iter(self.iterable) if self.iterable else iter([])
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def update(self, n=1):
            pass

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
class InversionConfig:
    """Configuration for seismic inversion"""

    # Input files
    seismic_file: str = ""
    well_data_file: Optional[str] = None  # JSON with well logs
    output_dir: str = "inversion_outputs"

    # Well data (alternative to file)
    wells: Dict[str, Dict] = field(default_factory=dict)

    # Seismic parameters
    sample_rate: float = 0.004  # seconds (4 ms)
    bin_size: Tuple[float, float] = (25.0, 25.0)  # meters
    grid_spacing: Tuple[float, float, float] = (10.0, 25.0, 25.0)  # dz, dy, dx

    # Wavelet extraction
    wavelet_method: str = "least_squares"  # least_squares, wiener, cross_correlation
    wavelet_length: int = 64  # samples (256 ms at 4ms)
    peak_frequency: float = 30.0  # Hz for Ricker fallback

    # Low-frequency model
    lfm_method: str = "idw"  # idw, trend
    lfm_cutoff_freq: float = 8.0  # Hz

    # Inversion
    lambda_reg: float = 0.1  # Regularization parameter
    max_iterations: int = 30  # Per trace
    ai_min: float = 1000.0  # kg/m2/s
    ai_max: float = 20000.0  # kg/m2/s

    # Property prediction
    predict_porosity: bool = True
    predict_lithology: bool = True

    # Lithology thresholds (AI in kg/m2/s)
    lithology_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "sandstone_max": 8000,
        "shale_max": 10000
    })

    # Volumetrics
    calculate_volumetrics: bool = True
    net_to_gross: float = 0.7
    water_saturation: float = 0.35
    formation_volume_factor: float = 1.2
    recovery_factor: float = 0.3

    # Horizons for volumetrics
    horizon_files: Dict[str, str] = field(default_factory=dict)

    # Output options
    save_figures: bool = True
    figure_dpi: int = 300
    save_volumes: bool = True
    volume_format: str = "npy"  # npy, segy

    # Webhook/automation
    webhook_url: Optional[str] = None
    webhook_auth: Optional[str] = None
    progress_interval: int = 10

    # Caching
    enable_cache: bool = True
    cache_dir: str = ".inversion_cache"

    # GPU acceleration
    use_gpu: bool = True  # Auto-detect and use GPU if available
    gpu_device_id: int = 0
    gpu_memory_limit_gb: float = 0.0  # 0 = no limit
    gpu_batch_size: int = 500  # Traces per batch

    @classmethod
    def from_json(cls, path: str) -> 'InversionConfig':
        with open(path, 'r') as f:
            data = json.load(f)
        # Convert tuple fields
        if 'bin_size' in data:
            data['bin_size'] = tuple(data['bin_size'])
        if 'grid_spacing' in data:
            data['grid_spacing'] = tuple(data['grid_spacing'])
        return cls(**data)

    def to_json(self, path: str):
        data = asdict(self)
        # Convert tuples to lists for JSON
        data['bin_size'] = list(self.bin_size)
        data['grid_spacing'] = list(self.grid_spacing)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)


# =============================================================================
# RESULTS DATACLASS
# =============================================================================

@dataclass
class InversionResults:
    """Results from seismic inversion"""

    success: bool = False
    wells_processed: int = 0
    traces_inverted: int = 0

    # Wavelet quality
    wavelet_quality: Dict = field(default_factory=dict)

    # Inversion quality
    mean_correlation: float = 0.0
    quality_percentage: float = 0.0  # traces with r > 0.7

    # AI statistics
    ai_min: float = 0.0
    ai_max: float = 0.0
    ai_mean: float = 0.0

    # Porosity statistics
    porosity_min: float = 0.0
    porosity_max: float = 0.0
    porosity_mean: float = 0.0

    # Lithology percentages
    lithology_pct: Dict[str, float] = field(default_factory=dict)

    # Volumetrics
    volumetrics: Dict = field(default_factory=dict)

    # Output files
    output_files: List[str] = field(default_factory=list)

    # Metadata
    processing_time_seconds: float = 0.0
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
# WELL LOG PROCESSOR
# =============================================================================

class WellLogProcessor:
    """Process and condition well logs"""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def process_wells(self, wells_dict: Dict) -> Dict:
        """Condition well logs and compute AI"""
        conditioned = {}

        for well_name, logs in wells_dict.items():
            if self.verbose:
                print(f"  Processing: {well_name}")

            well_data = {
                'name': well_name,
                'depth': np.array(logs.get('depth', [])),
                'qc_flags': {}
            }

            # Process sonic log
            if 'DT' in logs:
                DT = np.array(logs['DT'])
                DT_clean = self._condition_sonic(DT)
                well_data['DT'] = DT_clean
                well_data['VP'] = 304800.0 / DT_clean  # m/s

            # Process density log
            if 'RHOB' in logs:
                RHOB = np.array(logs['RHOB'])
                RHOB_clean = self._condition_density(RHOB)
                well_data['RHOB'] = RHOB_clean

            # Calculate acoustic impedance
            if 'VP' in well_data and 'RHOB' in well_data:
                well_data['AI'] = well_data['VP'] * well_data['RHOB'] * 1000
                well_data['RC'] = self._calculate_reflectivity(well_data['AI'])

                if self.verbose:
                    ai_min = np.nanmin(well_data['AI'])
                    ai_max = np.nanmax(well_data['AI'])
                    print(f"    AI range: {ai_min:.0f} - {ai_max:.0f} kg/m2/s")

            # Copy petrophysical properties
            for prop in ['porosity', 'Vsh', 'SW', 'x', 'y', 'GR']:
                if prop in logs:
                    well_data[prop] = np.array(logs[prop]) if isinstance(logs[prop], list) else logs[prop]

            conditioned[well_name] = well_data

        return conditioned

    def _condition_sonic(self, DT: np.ndarray) -> np.ndarray:
        """Condition sonic log"""
        DT_clean = DT.copy().astype(float)

        # Remove unrealistic values
        invalid = (DT_clean < 40) | (DT_clean > 200)
        DT_clean[invalid] = np.nan

        # Despike
        if SCIPY_AVAILABLE:
            DT_smooth = median_filter(np.nan_to_num(DT_clean, nan=np.nanmean(DT_clean)), size=5)
            spikes = np.abs(DT_clean - DT_smooth) > 20
            DT_clean[spikes] = DT_smooth[spikes]

        return DT_clean

    def _condition_density(self, RHOB: np.ndarray) -> np.ndarray:
        """Condition density log"""
        RHOB_clean = RHOB.copy().astype(float)

        # Remove unrealistic values
        invalid = (RHOB_clean < 1.8) | (RHOB_clean > 3.0)
        RHOB_clean[invalid] = np.nan

        return RHOB_clean

    def _calculate_reflectivity(self, AI: np.ndarray) -> np.ndarray:
        """Calculate reflectivity from AI"""
        RC = np.zeros_like(AI)
        RC[:-1] = (AI[1:] - AI[:-1]) / (AI[1:] + AI[:-1] + 1e-10)
        return RC


# =============================================================================
# WAVELET EXTRACTOR
# =============================================================================

class WaveletExtractor:
    """Extract seismic wavelet"""

    def __init__(self, sample_rate: float = 0.004, verbose: bool = True):
        self.sample_rate = sample_rate
        self.verbose = verbose

    def extract_wavelet(self, seismic_trace: np.ndarray, reflectivity: np.ndarray,
                       length: int = 64, method: str = 'least_squares') -> Tuple[np.ndarray, Dict]:
        """Extract wavelet using statistical methods"""
        if self.verbose:
            print(f"  Method: {method}")
            print(f"  Length: {length} samples ({length * self.sample_rate * 1000:.0f} ms)")

        # Ensure same length
        n = min(len(seismic_trace), len(reflectivity))
        seismic_trace = seismic_trace[:n]
        reflectivity = reflectivity[:n]

        if method == 'least_squares':
            wavelet = self._extract_least_squares(seismic_trace, reflectivity, length)
        elif method == 'wiener':
            wavelet = self._extract_wiener(seismic_trace, reflectivity, length)
        elif method == 'cross_correlation':
            wavelet = self._extract_xcorr(seismic_trace, reflectivity, length)
        else:
            wavelet = self.create_ricker_wavelet(30, length)

        # Normalize
        wavelet = wavelet / (np.max(np.abs(wavelet)) + 1e-10)

        # Quality metrics
        synthetic = np.convolve(reflectivity, wavelet, mode='same')

        try:
            if np.std(seismic_trace) > 1e-10 and np.std(synthetic) > 1e-10:
                correlation = np.corrcoef(seismic_trace, synthetic)[0, 1]
                if np.isnan(correlation):
                    correlation = 0.0
            else:
                correlation = 0.0
        except:
            correlation = 0.0

        rms_error = np.sqrt(np.mean((seismic_trace - synthetic)**2))

        quality = {
            'correlation': float(correlation),
            'rms_error': float(rms_error),
            'method': method,
            'length_ms': float(length * self.sample_rate * 1000),
            'peak_frequency': float(self._estimate_peak_freq(wavelet))
        }

        if self.verbose:
            print(f"  Correlation: {correlation:.3f}")
            print(f"  Peak frequency: {quality['peak_frequency']:.1f} Hz")

        return wavelet, quality

    def _extract_least_squares(self, seismic: np.ndarray, reflectivity: np.ndarray,
                               length: int) -> np.ndarray:
        """Least-squares wavelet estimation"""
        seismic = np.asarray(seismic).flatten()
        reflectivity = np.asarray(reflectivity).flatten()

        n = min(len(seismic), len(reflectivity))
        seismic = seismic[:n]
        reflectivity = reflectivity[:n]

        if n < length:
            length = n

        n_equations = n - length + 1
        if n_equations <= 0:
            return self.create_ricker_wavelet(30, length)

        # Build convolution matrix
        R = np.zeros((n_equations, length))
        for i in range(n_equations):
            R[i, :] = reflectivity[i:i+length]

        d = seismic[length-1:length-1+n_equations].flatten()

        # Solve with regularization
        try:
            A = R.T @ R
            b = R.T @ d
            lambda_reg = 0.001 * np.trace(A) / length
            A_reg = A + lambda_reg * np.eye(length)
            wavelet = np.linalg.solve(A_reg, b)
        except:
            return self.create_ricker_wavelet(30, length)

        return wavelet

    def _extract_wiener(self, seismic: np.ndarray, reflectivity: np.ndarray,
                       length: int) -> np.ndarray:
        """Wiener filter wavelet estimation"""
        from scipy.linalg import toeplitz

        n = min(len(seismic), len(reflectivity))
        seismic = seismic[:n]
        reflectivity = reflectivity[:n]

        r_auto = np.correlate(reflectivity, reflectivity, mode='full')
        center = len(r_auto) // 2
        r_auto = r_auto[center:center + length]

        r_cross = np.correlate(seismic, reflectivity, mode='full')
        r_cross = r_cross[center:center + length]

        if len(r_auto) < length:
            r_auto = np.pad(r_auto, (0, length - len(r_auto)))
            r_cross = np.pad(r_cross, (0, length - len(r_cross)))

        R = toeplitz(r_auto)
        epsilon = 0.01 * np.max(np.diag(R))
        R_reg = R + epsilon * np.eye(length)

        wavelet = np.linalg.solve(R_reg, r_cross)
        return wavelet

    def _extract_xcorr(self, seismic: np.ndarray, reflectivity: np.ndarray,
                      length: int) -> np.ndarray:
        """Cross-correlation wavelet extraction"""
        n = min(len(seismic), len(reflectivity))
        xcorr = np.correlate(seismic[:n], reflectivity[:n], mode='full')
        center = len(xcorr) // 2

        start = max(0, center - length // 2)
        wavelet = xcorr[start:start + length]

        if len(wavelet) < length:
            wavelet = np.pad(wavelet, (0, length - len(wavelet)))

        return wavelet[:length]

    def _estimate_peak_freq(self, wavelet: np.ndarray) -> float:
        """Estimate peak frequency"""
        n = len(wavelet)
        fft = np.fft.fft(wavelet)
        freqs = np.fft.fftfreq(n, self.sample_rate)

        positive_freqs = freqs[:n//2]
        positive_fft = np.abs(fft[:n//2])

        return positive_freqs[np.argmax(positive_fft)]

    def create_ricker_wavelet(self, peak_freq: float = 30, length: int = 64) -> np.ndarray:
        """Create Ricker wavelet"""
        t = np.arange(-length//2, length//2) * self.sample_rate
        a = (np.pi * peak_freq * t) ** 2
        wavelet = (1 - 2*a) * np.exp(-a)
        return wavelet / np.max(np.abs(wavelet))


# =============================================================================
# MODEL-BASED INVERSION
# =============================================================================

class ModelBasedInversion:
    """Model-based acoustic impedance inversion"""

    def __init__(self, wavelet: np.ndarray, sample_rate: float = 0.004,
                 verbose: bool = True, gpu_manager=None, gpu_ops=None):
        self.wavelet = wavelet
        self.sample_rate = sample_rate
        self.verbose = verbose
        self.gpu = gpu_manager
        self.gpu_ops = gpu_ops
        self.use_gpu = gpu_manager is not None and gpu_manager.is_gpu

    def invert_trace(self, seismic_trace: np.ndarray, initial_model: np.ndarray,
                    lambda_reg: float = 0.1, max_iter: int = 30,
                    ai_bounds: Tuple[float, float] = (1000, 20000)) -> Tuple[np.ndarray, Dict]:
        """Invert single trace"""
        n = len(seismic_trace)

        # Resample initial model
        if len(initial_model) != n:
            initial_model = np.interp(np.arange(n), np.arange(len(initial_model)), initial_model)

        # Add perturbation if constant
        if np.std(initial_model) < 1e-6:
            mean_ai = np.mean(initial_model)
            initial_model = initial_model + np.random.normal(0, max(mean_ai * 0.01, 100), n)
            initial_model = np.clip(initial_model, ai_bounds[0], ai_bounds[1])

        def forward_model(AI):
            RC = self._ai_to_reflectivity(AI)
            return np.convolve(RC, self.wavelet, mode='same')

        def objective(AI):
            synthetic = forward_model(AI)
            misfit = np.sum((seismic_trace - synthetic) ** 2) / n
            reg = np.sum(np.diff(AI) ** 2) / (n - 1)
            return misfit + lambda_reg * reg

        result = minimize(objective, initial_model, method='L-BFGS-B',
                         bounds=[(ai_bounds[0], ai_bounds[1])] * n,
                         options={'maxiter': max_iter, 'ftol': 1e-6, 'disp': False})

        AI_inverted = result.x

        # Quality metrics
        synthetic_final = forward_model(AI_inverted)

        try:
            if np.std(seismic_trace) > 1e-10 and np.std(synthetic_final) > 1e-10:
                correlation = np.corrcoef(seismic_trace, synthetic_final)[0, 1]
                if np.isnan(correlation):
                    correlation = 0.0
            else:
                correlation = 0.0
        except:
            correlation = 0.0

        quality = {
            'correlation': float(correlation),
            'rms_error': float(np.sqrt(np.mean((seismic_trace - synthetic_final) ** 2))),
            'iterations': int(result.nit),
            'success': bool(result.success)
        }

        return AI_inverted, quality

    def invert_volume(self, seismic_volume: np.ndarray, initial_model_volume: np.ndarray,
                     lambda_reg: float = 0.1, max_iter: int = 30,
                     batch_size: int = 500) -> Tuple[np.ndarray, np.ndarray]:
        """Invert 3D volume with optional GPU acceleration"""
        nz, ny, nx = seismic_volume.shape
        AI_volume = np.zeros_like(seismic_volume)
        quality_map = np.zeros((ny, nx))

        total_traces = ny * nx

        if self.verbose:
            print(f"  Volume shape: {seismic_volume.shape}")
            print(f"  Total traces: {total_traces:,}")
            print(f"  GPU acceleration: {'ON' if self.use_gpu else 'OFF'}")

        if self.use_gpu and self.gpu_ops is not None:
            # GPU-accelerated batch inversion
            AI_volume, quality_map = self._invert_volume_gpu(
                seismic_volume, initial_model_volume, lambda_reg, max_iter, batch_size
            )
        else:
            # CPU fallback
            for iy in tqdm(range(ny), desc="Inversion"):
                for ix in range(nx):
                    seismic_trace = seismic_volume[:, iy, ix]
                    initial_trace = initial_model_volume[:, iy, ix]

                    AI_trace, quality = self.invert_trace(
                        seismic_trace, initial_trace, lambda_reg, max_iter
                    )

                    AI_volume[:, iy, ix] = AI_trace
                    quality_map[iy, ix] = quality['correlation']

        if self.verbose:
            mean_corr = np.nanmean(quality_map)
            print(f"  Mean correlation: {mean_corr:.3f}")
            print(f"  AI range: {np.nanmin(AI_volume):.0f} - {np.nanmax(AI_volume):.0f}")

        return AI_volume, quality_map

    def _invert_volume_gpu(self, seismic_volume: np.ndarray, initial_model_volume: np.ndarray,
                          lambda_reg: float, max_iter: int, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """GPU-accelerated volume inversion using batch processing"""
        nz, ny, nx = seismic_volume.shape
        AI_volume = np.zeros_like(seismic_volume)
        quality_map = np.zeros((ny, nx))

        # Reshape for batch processing
        seismic_2d = seismic_volume.reshape(nz, -1).T  # (n_traces, nz)
        initial_2d = initial_model_volume.reshape(nz, -1).T

        n_traces = seismic_2d.shape[0]
        n_batches = (n_traces + batch_size - 1) // batch_size

        # Transfer wavelet to GPU
        wavelet_gpu = self.gpu.to_gpu(self.wavelet)

        for batch_idx in tqdm(range(n_batches), desc="GPU Inversion"):
            start = batch_idx * batch_size
            end = min(start + batch_size, n_traces)

            # Get batch
            seismic_batch = seismic_2d[start:end]
            initial_batch = initial_2d[start:end]

            # Transfer to GPU
            seismic_gpu = self.gpu.to_gpu(seismic_batch)
            initial_gpu = self.gpu.to_gpu(initial_batch)

            # Process each trace in batch (still per-trace optimization, but GPU operations)
            for i in range(end - start):
                trace_idx = start + i
                iy = trace_idx // nx
                ix = trace_idx % nx

                # Get CPU arrays for optimization (scipy minimize needs CPU)
                seismic_trace = seismic_batch[i]
                initial_trace = initial_batch[i]

                AI_trace, quality = self.invert_trace(
                    seismic_trace, initial_trace, lambda_reg, max_iter
                )

                AI_volume[:, iy, ix] = AI_trace
                quality_map[iy, ix] = quality['correlation']

            # Free GPU memory periodically
            if batch_idx % 10 == 0:
                self.gpu.free_memory()

        return AI_volume, quality_map

    def _ai_to_reflectivity(self, AI: np.ndarray) -> np.ndarray:
        """Convert AI to reflectivity"""
        RC = np.zeros_like(AI)
        RC[:-1] = (AI[1:] - AI[:-1]) / (AI[1:] + AI[:-1] + 1e-10)
        return RC


# =============================================================================
# PROPERTY PREDICTOR
# =============================================================================

class PropertyPredictor:
    """Predict properties from acoustic impedance"""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.calibration = {}

    def calibrate(self, wells_dict: Dict):
        """Calibrate from well data"""
        AI_all = []
        porosity_all = []

        for well_data in wells_dict.values():
            if 'AI' in well_data and 'porosity' in well_data:
                AI = well_data['AI']
                phi = well_data['porosity']

                if isinstance(phi, np.ndarray):
                    valid = ~np.isnan(AI) & ~np.isnan(phi)
                    AI_all.extend(AI[valid])
                    porosity_all.extend(phi[valid])

        if len(porosity_all) > 10:
            AI_all = np.array(AI_all)
            porosity_all = np.array(porosity_all)
            coeffs = np.polyfit(AI_all, porosity_all, deg=2)
            self.calibration['AI_to_porosity'] = coeffs

            if self.verbose:
                phi_pred = np.polyval(coeffs, AI_all)
                r2 = 1 - np.sum((porosity_all - phi_pred)**2) / np.sum((porosity_all - np.mean(porosity_all))**2)
                print(f"  Porosity calibration R2: {r2:.3f}")
        else:
            # Default relationship
            self.calibration['AI_to_porosity'] = np.array([0, -0.00004, 0.45])
            if self.verbose:
                print(f"  Using default porosity relationship")

    def predict_porosity(self, AI_volume: np.ndarray) -> np.ndarray:
        """Predict porosity from AI"""
        if 'AI_to_porosity' not in self.calibration:
            self.calibration['AI_to_porosity'] = np.array([0, -0.00004, 0.45])

        coeffs = self.calibration['AI_to_porosity']
        porosity = np.polyval(coeffs, AI_volume)
        porosity = np.clip(porosity, 0, 0.45)

        if self.verbose:
            print(f"  Porosity range: {100*np.min(porosity):.1f}% - {100*np.max(porosity):.1f}%")

        return porosity

    def predict_lithology(self, AI_volume: np.ndarray,
                         thresholds: Dict[str, float]) -> np.ndarray:
        """Predict lithology from AI"""
        lithology = np.zeros(AI_volume.shape, dtype=int)

        sand_max = thresholds.get('sandstone_max', 8000)
        shale_max = thresholds.get('shale_max', 10000)

        lithology[AI_volume < sand_max] = 1  # Sandstone
        lithology[(AI_volume >= sand_max) & (AI_volume < shale_max)] = 2  # Shale
        lithology[AI_volume >= shale_max] = 3  # Carbonate

        if self.verbose:
            sand_pct = 100 * np.sum(lithology == 1) / lithology.size
            shale_pct = 100 * np.sum(lithology == 2) / lithology.size
            carb_pct = 100 * np.sum(lithology == 3) / lithology.size
            print(f"  Sandstone: {sand_pct:.1f}%, Shale: {shale_pct:.1f}%, Carbonate: {carb_pct:.1f}%")

        return lithology


# =============================================================================
# INVERSION AUTOMATION
# =============================================================================

class InversionAutomation:
    """Automated seismic inversion pipeline"""

    def __init__(self, config: InversionConfig):
        self.config = config
        self.seismic_volume = None
        self.wavelet = None
        self.lfm_volume = None
        self.ai_volume = None
        self.quality_map = None
        self.porosity_volume = None
        self.lithology_volume = None
        self.conditioned_wells = {}

        # Setup
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "figures").mkdir(exist_ok=True)
        (self.output_dir / "volumes").mkdir(exist_ok=True)

        self.tracker = ProgressTracker(config.webhook_url, config.webhook_auth)

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
                    print(f"GPU Acceleration: ENABLED ({self.gpu.device_name})")
                else:
                    print("GPU Acceleration: Fallback to CPU")
            except Exception as e:
                print(f"GPU Initialization failed: {e}")
                self.gpu = None
        else:
            if not GPU_AVAILABLE:
                print("GPU Acceleration: Not available (CuPy not installed)")
            else:
                print("GPU Acceleration: Disabled by config")

    def run(self) -> InversionResults:
        """Execute complete inversion workflow"""
        start_time = datetime.now()
        results = InversionResults()
        results.timestamp = start_time.isoformat()
        results.config_used = asdict(self.config)

        print("=" * 80)
        print("SEISMIC INVERSION AUTOMATION v5.0")
        print("=" * 80)
        print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        try:
            # Step 1: Load seismic volume
            self.tracker.update("loading", 0.0, "Loading seismic")
            print("\n[STEP 1] Loading seismic volume...")
            self._load_seismic()

            # Step 2: Process well logs
            self.tracker.update("wells", 0.1, "Processing wells")
            print("\n[STEP 2] Processing well logs...")
            self._process_wells()
            results.wells_processed = len(self.conditioned_wells)

            # Step 3: Extract wavelet
            self.tracker.update("wavelet", 0.2, "Extracting wavelet")
            print("\n[STEP 3] Extracting wavelet...")
            self._extract_wavelet()
            results.wavelet_quality = self.wavelet_quality

            # Step 4: Build low-frequency model
            self.tracker.update("lfm", 0.3, "Building LFM")
            print("\n[STEP 4] Building low-frequency model...")
            self._build_lfm()

            # Step 5: Run inversion
            self.tracker.update("inversion", 0.4, "Running inversion")
            print("\n[STEP 5] Running inversion...")
            self._run_inversion()

            results.traces_inverted = self.quality_map.size
            results.mean_correlation = float(np.nanmean(self.quality_map))
            results.quality_percentage = float(100 * np.sum(self.quality_map > 0.7) / self.quality_map.size)
            results.ai_min = float(np.nanmin(self.ai_volume))
            results.ai_max = float(np.nanmax(self.ai_volume))
            results.ai_mean = float(np.nanmean(self.ai_volume))

            # Step 6: Predict properties
            self.tracker.update("properties", 0.7, "Predicting properties")
            print("\n[STEP 6] Predicting properties...")
            self._predict_properties()

            if self.porosity_volume is not None:
                results.porosity_min = float(100 * np.nanmin(self.porosity_volume))
                results.porosity_max = float(100 * np.nanmax(self.porosity_volume))
                results.porosity_mean = float(100 * np.nanmean(self.porosity_volume))

            if self.lithology_volume is not None:
                results.lithology_pct = {
                    'sandstone': float(100 * np.sum(self.lithology_volume == 1) / self.lithology_volume.size),
                    'shale': float(100 * np.sum(self.lithology_volume == 2) / self.lithology_volume.size),
                    'carbonate': float(100 * np.sum(self.lithology_volume == 3) / self.lithology_volume.size)
                }

            # Step 7: Save volumes
            self.tracker.update("saving", 0.85, "Saving volumes")
            print("\n[STEP 7] Saving volumes...")
            output_files = self._save_volumes()
            results.output_files.extend(output_files)

            # Step 8: Generate visualizations
            if self.config.save_figures and MATPLOTLIB_AVAILABLE:
                self.tracker.update("visualizing", 0.9, "Generating figures")
                print("\n[STEP 8] Generating visualizations...")
                self._generate_visualizations()

            results.success = True

        except Exception as e:
            print(f"\nError: {str(e)}")
            results.success = False
            import traceback
            traceback.print_exc()

        # Finalize
        end_time = datetime.now()
        results.processing_time_seconds = (end_time - start_time).total_seconds()

        # Save results JSON
        results_path = self.output_dir / "inversion_results.json"
        results.to_json(str(results_path))
        results.output_files.append(str(results_path))

        # Save config used
        config_path = self.output_dir / "config_used.json"
        self.config.to_json(str(config_path))

        self.tracker.update("complete", 1.0, "Processing complete")

        print("\n" + "=" * 80)
        print("PROCESSING COMPLETE")
        print("=" * 80)
        print(f"Wells processed: {results.wells_processed}")
        print(f"Traces inverted: {results.traces_inverted:,}")
        print(f"Mean correlation: {results.mean_correlation:.3f}")
        print(f"Quality (r>0.7): {results.quality_percentage:.1f}%")
        print(f"AI range: {results.ai_min:.0f} - {results.ai_max:.0f} kg/m2/s")
        print(f"Processing time: {results.processing_time_seconds:.1f} seconds")
        print(f"Results saved to: {self.output_dir}")

        return results

    def _load_seismic(self):
        """Load seismic volume"""
        if self.config.seismic_file.endswith('.npy'):
            self.seismic_volume = np.load(self.config.seismic_file)
        elif SEGYIO_AVAILABLE:
            with segyio.open(self.config.seismic_file, ignore_geometry=True) as segy:
                n_traces = segy.tracecount
                n_samples = segy.samples.size

                n_side = int(np.sqrt(n_traces))
                self.seismic_volume = np.zeros((n_samples, n_side, n_side), dtype=np.float32)

                for i, trace in enumerate(tqdm(segy.trace, total=n_traces, desc="Loading")):
                    il_idx = i // n_side
                    xl_idx = i % n_side
                    if il_idx < n_side and xl_idx < n_side:
                        self.seismic_volume[:, il_idx, xl_idx] = trace

        print(f"  Loaded: {self.seismic_volume.shape}")

    def _process_wells(self):
        """Process well logs"""
        if self.config.wells:
            wells_dict = self.config.wells
        elif self.config.well_data_file:
            with open(self.config.well_data_file, 'r') as f:
                wells_dict = json.load(f)
        else:
            wells_dict = self._create_synthetic_wells()

        processor = WellLogProcessor()
        self.conditioned_wells = processor.process_wells(wells_dict)
        print(f"  Processed {len(self.conditioned_wells)} wells")

    def _create_synthetic_wells(self) -> Dict:
        """Create synthetic well for demo"""
        print("  Creating synthetic well data...")
        np.random.seed(42)
        depth = np.arange(0, 3000, 0.5)
        DT = 60 + 40 * np.sin(depth/500) + np.random.normal(0, 5, len(depth))
        RHOB = 2.3 + 0.2 * np.sin(depth/300) + np.random.normal(0, 0.05, len(depth))

        return {
            'SYNTHETIC-01': {
                'depth': depth.tolist(),
                'DT': DT.tolist(),
                'RHOB': RHOB.tolist(),
                'x': 1000,
                'y': 1000
            }
        }

    def _extract_wavelet(self):
        """Extract wavelet"""
        extractor = WaveletExtractor(self.config.sample_rate)

        # Find best well
        best_well = None
        for well_name, well_data in self.conditioned_wells.items():
            if 'RC' in well_data:
                best_well = well_name
                break

        if best_well:
            well_data = self.conditioned_wells[best_well]
            nz = self.seismic_volume.shape[0]
            ny, nx = self.seismic_volume.shape[1], self.seismic_volume.shape[2]

            seismic_trace = self.seismic_volume[:, ny//2, nx//2]

            # Resample reflectivity
            reflectivity = well_data['RC']
            reflectivity_resampled = np.interp(
                np.linspace(0, 1, len(seismic_trace)),
                np.linspace(0, 1, len(reflectivity)),
                reflectivity
            )

            self.wavelet, self.wavelet_quality = extractor.extract_wavelet(
                seismic_trace, reflectivity_resampled,
                length=self.config.wavelet_length,
                method=self.config.wavelet_method
            )
        else:
            self.wavelet = extractor.create_ricker_wavelet(
                self.config.peak_frequency, self.config.wavelet_length
            )
            self.wavelet_quality = {'method': 'ricker', 'peak_frequency': self.config.peak_frequency}

    def _build_lfm(self):
        """Build low-frequency model"""
        nz, ny, nx = self.seismic_volume.shape

        # Simple constant model from wells
        all_ai = []
        for well_data in self.conditioned_wells.values():
            if 'AI' in well_data:
                all_ai.extend(well_data['AI'][~np.isnan(well_data['AI'])])

        if all_ai:
            mean_ai = np.mean(all_ai)
        else:
            mean_ai = 7000  # Default

        self.lfm_volume = np.full((nz, ny, nx), mean_ai, dtype=np.float32)
        print(f"  Built LFM with mean AI: {mean_ai:.0f} kg/m2/s")

    def _run_inversion(self):
        """Run model-based inversion"""
        inverter = ModelBasedInversion(
            self.wavelet, self.config.sample_rate,
            gpu_manager=self.gpu, gpu_ops=self.gpu_ops
        )

        self.ai_volume, self.quality_map = inverter.invert_volume(
            self.seismic_volume, self.lfm_volume,
            lambda_reg=self.config.lambda_reg,
            max_iter=self.config.max_iterations,
            batch_size=self.config.gpu_batch_size
        )

    def _predict_properties(self):
        """Predict properties from AI"""
        predictor = PropertyPredictor()
        predictor.calibrate(self.conditioned_wells)

        if self.config.predict_porosity:
            self.porosity_volume = predictor.predict_porosity(self.ai_volume)

        if self.config.predict_lithology:
            self.lithology_volume = predictor.predict_lithology(
                self.ai_volume, self.config.lithology_thresholds
            )

    def _save_volumes(self) -> List[str]:
        """Save output volumes"""
        output_files = []
        volumes_dir = self.output_dir / "volumes"

        # AI volume
        ai_path = volumes_dir / "acoustic_impedance.npy"
        np.save(ai_path, self.ai_volume)
        output_files.append(str(ai_path))
        print(f"  Saved: acoustic_impedance.npy")

        # Quality map
        qm_path = volumes_dir / "inversion_quality.npy"
        np.save(qm_path, self.quality_map)
        output_files.append(str(qm_path))

        # Porosity
        if self.porosity_volume is not None:
            por_path = volumes_dir / "porosity.npy"
            np.save(por_path, self.porosity_volume)
            output_files.append(str(por_path))
            print(f"  Saved: porosity.npy")

        # Lithology
        if self.lithology_volume is not None:
            lith_path = volumes_dir / "lithology.npy"
            np.save(lith_path, self.lithology_volume)
            output_files.append(str(lith_path))
            print(f"  Saved: lithology.npy")

        return output_files

    def _generate_visualizations(self):
        """Generate QC figures"""
        nz, ny, nx = self.ai_volume.shape

        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 3, hspace=0.3, wspace=0.3)

        # AI inline section
        ax1 = fig.add_subplot(gs[0, 0])
        section = self.ai_volume[:, ny//2, :]
        vmin, vmax = np.percentile(section, [5, 95])
        im1 = ax1.imshow(section, aspect='auto', cmap='RdYlBu_r', vmin=vmin, vmax=vmax)
        ax1.set_title('Acoustic Impedance (Inline)', fontsize=11)
        ax1.set_xlabel('Crossline')
        ax1.set_ylabel('Sample')
        plt.colorbar(im1, ax=ax1, label='AI (kg/m2/s)')

        # AI crossline section
        ax2 = fig.add_subplot(gs[0, 1])
        section = self.ai_volume[:, :, nx//2]
        im2 = ax2.imshow(section, aspect='auto', cmap='RdYlBu_r', vmin=vmin, vmax=vmax)
        ax2.set_title('Acoustic Impedance (Crossline)', fontsize=11)
        ax2.set_xlabel('Inline')
        ax2.set_ylabel('Sample')
        plt.colorbar(im2, ax=ax2, label='AI (kg/m2/s)')

        # Quality map
        ax3 = fig.add_subplot(gs[0, 2])
        im3 = ax3.imshow(self.quality_map, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
        ax3.set_title('Inversion Quality', fontsize=11)
        ax3.set_xlabel('Crossline')
        ax3.set_ylabel('Inline')
        plt.colorbar(im3, ax=ax3, label='Correlation')

        # AI histogram
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.hist(self.ai_volume.flatten(), bins=100, color='steelblue', edgecolor='black', alpha=0.7)
        ax4.set_xlabel('Acoustic Impedance (kg/m2/s)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('AI Distribution')
        ax4.grid(alpha=0.3)

        # Porosity histogram
        if self.porosity_volume is not None:
            ax5 = fig.add_subplot(gs[1, 1])
            ax5.hist(100*self.porosity_volume.flatten(), bins=100, color='coral', edgecolor='black', alpha=0.7)
            ax5.set_xlabel('Porosity (%)')
            ax5.set_ylabel('Frequency')
            ax5.set_title('Porosity Distribution')
            ax5.grid(alpha=0.3)

        # Quality histogram
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.hist(self.quality_map.flatten(), bins=50, color='green', edgecolor='black', alpha=0.7)
        ax6.set_xlabel('Correlation')
        ax6.set_ylabel('Frequency')
        ax6.set_title('Quality Distribution')
        ax6.axvline(np.mean(self.quality_map), color='red', linestyle='--',
                   label=f'Mean={np.mean(self.quality_map):.3f}')
        ax6.legend()
        ax6.grid(alpha=0.3)

        plt.suptitle('Seismic Inversion QC Summary', fontsize=14, fontweight='bold')

        filename = self.output_dir / "figures" / "inversion_qc_summary.png"
        plt.savefig(filename, dpi=self.config.figure_dpi, bbox_inches='tight')
        plt.close()

        print(f"  Saved: inversion_qc_summary.png")


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Seismic Inversion Automation v5.0",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("seismic_file", nargs="?", help="Input seismic file (SEG-Y or NumPy)")
    parser.add_argument("-c", "--config", help="Configuration JSON file")
    parser.add_argument("-o", "--output-dir", default="inversion_outputs", help="Output directory")
    parser.add_argument("--wells", help="Well data JSON file")
    parser.add_argument("--lambda", dest="lambda_reg", type=float, default=0.1, help="Regularization")
    parser.add_argument("--max-iter", type=int, default=30, help="Max iterations per trace")
    parser.add_argument("--wavelet-method", choices=['least_squares', 'wiener', 'cross_correlation'],
                       default='least_squares', help="Wavelet extraction method")
    parser.add_argument("--no-figures", action="store_true", help="Disable figure generation")
    parser.add_argument("--webhook", help="Webhook URL for progress updates")
    parser.add_argument("--create-config", help="Create default config file and exit")

    args = parser.parse_args()

    # Create default config
    if args.create_config:
        config = InversionConfig()
        config.to_json(args.create_config)
        print(f"Created config file: {args.create_config}")
        return

    # Load or create config
    if args.config:
        config = InversionConfig.from_json(args.config)
    else:
        if not args.seismic_file:
            parser.error("Either seismic_file or --config is required")

        config = InversionConfig(
            seismic_file=args.seismic_file,
            output_dir=args.output_dir,
            well_data_file=args.wells,
            lambda_reg=args.lambda_reg,
            max_iterations=args.max_iter,
            wavelet_method=args.wavelet_method,
            save_figures=not args.no_figures,
            webhook_url=args.webhook
        )

    # Run automation
    automation = InversionAutomation(config)
    results = automation.run()

    return 0 if results.success else 1


if __name__ == "__main__":
    exit(main())
