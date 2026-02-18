"""
================================================================================
AUTOMATED HORIZON INTERPRETATION v5.0
Production-ready region-growing horizon tracking for seismic volumes
================================================================================

Features:
- Region-growing horizon tracking with multi-scale seeding
- Formation-specific tracking parameters
- Structural closure identification
- JSON structured output for automation
- CLI, API, and webhook support
- Intelligent caching

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

# Optional imports with fallbacks
try:
    import segyio
    SEGYIO_AVAILABLE = True
except ImportError:
    SEGYIO_AVAILABLE = False
    print("Warning: segyio not available")

try:
    from scipy import ndimage
    from scipy.interpolate import griddata
    from scipy.signal import correlate
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available")

try:
    from skimage import measure, morphology
    from skimage.segmentation import watershed
    from skimage.feature import peak_local_max
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("Warning: skimage not available")

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
        def set_description(self, desc):
            pass

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class HorizonInterpretationConfig:
    """Configuration for horizon interpretation"""

    # Input files
    seismic_file: str = ""
    coherence_file: Optional[str] = None
    dip_file: Optional[str] = None
    output_dir: str = "horizon_outputs"

    # Horizons to track (name: [typical_time_ms, time_range_min, time_range_max, horizon_type, color])
    # Use generic names for publications - configure actual names in config file
    formations: Dict[str, List] = field(default_factory=lambda: {
        "Horizon_A": [500, 400, 600, "trough", "#FFD700"],
        "Horizon_B": [1200, 1000, 1400, "peak", "#FF8C00"],
        "Horizon_C": [2000, 1800, 2200, "peak", "#8B0000"],
        "Horizon_D": [2800, 2600, 3000, "trough", "#006400"],
        "Horizon_E": [3500, 3300, 3700, "peak", "#0000CD"],
        "Horizon_F": [4200, 4000, 4400, "peak", "#8B008B"]
    })

    # Well control (name: [lat, lon, TD, KB, GL])
    wells: Dict[str, Dict] = field(default_factory=dict)

    # Tracking parameters
    grid_density: int = 50  # Grid seed spacing in traces
    search_window: int = 30  # Samples to search for horizon
    similarity_threshold: float = 0.4  # Correlation threshold
    max_dip: float = 85.0  # Maximum apparent dip in degrees
    max_iterations: int = 200000  # Maximum tracking iterations
    correlation_window: int = 9  # Correlation window size in samples
    use_8_connected: bool = True  # 8 vs 4 connectivity

    # Fault-guided tracking
    fault_guided: bool = False
    coherence_threshold: float = 0.15

    # Velocity for dip calculation (m/s)
    avg_velocity: float = 2200.0
    trace_spacing: float = 25.0  # meters

    # Post-processing
    fill_small_gaps: bool = True
    max_gap_size: int = 5
    smooth_sigma: float = 1.5

    # Closure identification
    identify_closures: bool = True
    min_closure_amplitude: float = 100.0  # ms structural relief
    min_closure_area: int = 50  # traces
    contour_interval: float = 50.0  # ms

    # Output options
    save_figures: bool = True
    figure_dpi: int = 300
    export_horizons: bool = True
    export_format: str = "npy"  # npy, txt, or both

    # Webhook/automation
    webhook_url: Optional[str] = None
    webhook_auth: Optional[str] = None
    progress_interval: int = 10

    # Caching
    enable_cache: bool = True
    cache_dir: str = ".horizon_cache"

    @classmethod
    def from_json(cls, path: str) -> 'HorizonInterpretationConfig':
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
class HorizonResults:
    """Results from horizon interpretation"""

    success: bool = False
    horizons_tracked: int = 0
    total_closures: int = 0

    # Per-horizon results
    horizon_stats: Dict[str, Dict] = field(default_factory=dict)
    closures: Dict[str, List] = field(default_factory=dict)

    # QC metrics
    mean_coverage: float = 0.0
    mean_iterations: int = 0
    tracking_quality: str = "Unknown"

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
    """Track and report progress via webhook"""

    def __init__(self, webhook_url: Optional[str] = None,
                 webhook_auth: Optional[str] = None):
        self.webhook_url = webhook_url
        self.webhook_auth = webhook_auth
        self.start_time = datetime.now()

    def update(self, stage: str, progress: float, message: str = ""):
        """Send progress update"""
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
    """Cache tracking results for re-use"""

    def __init__(self, cache_dir: str, enabled: bool = True):
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled
        if enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_hash(self, config: HorizonInterpretationConfig, formation: str) -> str:
        """Generate hash from config and formation"""
        key_data = {
            'seismic_file': config.seismic_file,
            'formation': formation,
            'search_window': config.search_window,
            'similarity_threshold': config.similarity_threshold,
            'max_dip': config.max_dip,
            'grid_density': config.grid_density
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()[:12]

    def get(self, config: HorizonInterpretationConfig, formation: str) -> Optional[Dict]:
        """Retrieve cached result"""
        if not self.enabled:
            return None

        cache_file = self.cache_dir / f"horizon_{self._get_hash(config, formation)}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except:
                return None
        return None

    def set(self, config: HorizonInterpretationConfig, formation: str, data: Dict):
        """Store result in cache"""
        if not self.enabled:
            return

        cache_file = self.cache_dir / f"horizon_{self._get_hash(config, formation)}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
        except:
            pass


# =============================================================================
# HORIZON INTERPRETER
# =============================================================================

class HorizonInterpretationAutomation:
    """Automated horizon interpretation pipeline"""

    def __init__(self, config: HorizonInterpretationConfig):
        self.config = config
        self.volume = None
        self.coherence = None
        self.dip = None
        self.ilines = None
        self.xlines = None
        self.time = None
        self.sample_rate = None
        self.n_ilines = 0
        self.n_xlines = 0
        self.n_samples = 0
        self.horizons = {}
        self.diagnostics = {}

        # Setup
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "figures").mkdir(exist_ok=True)
        (self.output_dir / "data").mkdir(exist_ok=True)

        self.tracker = ProgressTracker(config.webhook_url, config.webhook_auth)
        self.cache = ResultCache(config.cache_dir, config.enable_cache)

    def run(self) -> HorizonResults:
        """Execute complete horizon interpretation workflow"""
        start_time = datetime.now()
        results = HorizonResults()
        results.timestamp = start_time.isoformat()
        results.config_used = asdict(self.config)

        print("=" * 80)
        print("HORIZON INTERPRETATION AUTOMATION v5.0")
        print("=" * 80)
        print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        try:
            # Step 1: Load seismic volume
            self.tracker.update("loading", 0.0, "Loading seismic volume")
            print("\n[STEP 1] Loading seismic volume...")
            self._load_seismic()

            # Step 2: Load attribute volumes if provided
            if self.config.coherence_file:
                print("[STEP 2] Loading coherence volume...")
                self._load_coherence()

            if self.config.dip_file:
                print("[STEP 3] Loading dip volume...")
                self._load_dip()

            # Step 3: Estimate well locations if provided
            if self.config.wells:
                print("[STEP 4] Estimating well locations...")
                self._estimate_well_locations()

            # Step 4: Track horizons
            print("\n[STEP 5] Tracking horizons...")
            self.tracker.update("tracking", 0.1, "Starting horizon tracking")

            formations = list(self.config.formations.keys())
            total_formations = len(formations)

            for idx, formation_name in enumerate(formations):
                progress = 0.1 + 0.7 * (idx / total_formations)
                self.tracker.update("tracking", progress, f"Tracking {formation_name}")

                print(f"\n{'=' * 60}")
                print(f"Processing: {formation_name}")
                print("=" * 60)

                # Check cache
                cached = self.cache.get(self.config, formation_name)
                if cached is not None:
                    print(f"  Using cached result")
                    self.horizons[formation_name] = cached
                    results.horizon_stats[formation_name] = cached.get('stats', {})
                    continue

                # Generate seeds
                seeds = self._generate_seeds(formation_name)

                # Track horizon
                formation_data = self.config.formations[formation_name]
                horizon_type = formation_data[3] if len(formation_data) > 3 else "peak"

                horizon_surface, stats = self._track_horizon(
                    formation_name, seeds, horizon_type
                )

                # Store results
                self.horizons[formation_name] = {
                    'surface': horizon_surface,
                    'stats': stats,
                    'horizon_type': horizon_type
                }
                results.horizon_stats[formation_name] = stats

                # Cache results
                self.cache.set(self.config, formation_name, self.horizons[formation_name])

                # Identify closures
                if self.config.identify_closures:
                    closures = self._identify_closures(formation_name, horizon_surface)
                    results.closures[formation_name] = closures
                    results.total_closures += len(closures)

                # Generate visualization
                if self.config.save_figures and MATPLOTLIB_AVAILABLE:
                    self._visualize_horizon(formation_name)
                    self._visualize_structure_map(formation_name)

            # Step 5: Export results
            print("\n[STEP 6] Exporting results...")
            self.tracker.update("exporting", 0.9, "Exporting results")

            if self.config.export_horizons:
                for formation_name in formations:
                    if formation_name in self.horizons:
                        surface = self.horizons[formation_name]['surface']

                        if self.config.export_format in ['npy', 'both']:
                            npy_path = self.output_dir / "data" / f"{formation_name}_surface.npy"
                            np.save(npy_path, surface)
                            results.output_files.append(str(npy_path))

                        if self.config.export_format in ['txt', 'both']:
                            txt_path = self.output_dir / "data" / f"{formation_name}_surface.txt"
                            np.savetxt(txt_path, surface, fmt='%.2f')
                            results.output_files.append(str(txt_path))

            # Calculate summary statistics
            coverages = [s.get('coverage', 0) for s in results.horizon_stats.values()]
            iterations = [s.get('iterations', 0) for s in results.horizon_stats.values()]

            results.horizons_tracked = len(self.horizons)
            results.mean_coverage = float(np.mean(coverages)) if coverages else 0.0
            results.mean_iterations = int(np.mean(iterations)) if iterations else 0

            if results.mean_coverage >= 60:
                results.tracking_quality = "Excellent"
            elif results.mean_coverage >= 40:
                results.tracking_quality = "Good"
            elif results.mean_coverage >= 20:
                results.tracking_quality = "Fair"
            else:
                results.tracking_quality = "Poor"

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
        results_path = self.output_dir / "horizon_results.json"
        results.to_json(str(results_path))
        results.output_files.append(str(results_path))

        # Save config used
        config_path = self.output_dir / "config_used.json"
        self.config.to_json(str(config_path))

        self.tracker.update("complete", 1.0, "Processing complete")

        print("\n" + "=" * 80)
        print("PROCESSING COMPLETE")
        print("=" * 80)
        print(f"Horizons tracked: {results.horizons_tracked}")
        print(f"Mean coverage: {results.mean_coverage:.1f}%")
        print(f"Total closures: {results.total_closures}")
        print(f"Quality: {results.tracking_quality}")
        print(f"Processing time: {results.processing_time_seconds:.1f} seconds")
        print(f"Results saved to: {self.output_dir}")

        return results

    def _load_seismic(self):
        """Load seismic volume"""
        if not SEGYIO_AVAILABLE:
            raise ImportError("segyio required for loading SEG-Y files")

        # Try with geometry
        try:
            with segyio.open(self.config.seismic_file, ignore_geometry=False) as segy:
                self.ilines = segy.ilines
                self.xlines = segy.xlines
                self.n_samples = segy.samples.size
                self.sample_rate = segyio.tools.dt(segy) / 1000
                self.volume = segyio.tools.cube(segy)
                self.time = np.arange(self.n_samples) * self.sample_rate
        except:
            # Try without geometry
            with segyio.open(self.config.seismic_file, ignore_geometry=True) as segy:
                n_traces = segy.tracecount
                self.n_samples = segy.samples.size
                self.sample_rate = segyio.tools.dt(segy) / 1000

                n_side = int(np.sqrt(n_traces))
                self.volume = np.zeros((n_side, n_side, self.n_samples), dtype=np.float32)

                for i, trace in enumerate(tqdm(segy.trace, total=n_traces, desc="Loading")):
                    il_idx = i // n_side
                    xl_idx = i % n_side
                    if il_idx < n_side and xl_idx < n_side:
                        self.volume[il_idx, xl_idx, :] = trace

                self.ilines = np.arange(n_side)
                self.xlines = np.arange(n_side)
                self.time = np.arange(self.n_samples) * self.sample_rate

        self.n_ilines, self.n_xlines, _ = self.volume.shape

        print(f"  Loaded: {self.n_ilines} x {self.n_xlines} x {self.n_samples}")
        print(f"  Time range: {self.time[0]:.0f} - {self.time[-1]:.0f} ms")
        print(f"  Sample rate: {self.sample_rate} ms")

    def _load_coherence(self):
        """Load coherence volume"""
        if os.path.exists(self.config.coherence_file):
            expected_shape = (self.n_ilines, self.n_xlines, self.n_samples)
            self.coherence = np.memmap(
                self.config.coherence_file, dtype=np.float32,
                mode='r', shape=expected_shape
            )
            print(f"  Loaded coherence: {expected_shape}")

    def _load_dip(self):
        """Load dip volume"""
        if os.path.exists(self.config.dip_file):
            expected_shape = (self.n_ilines, self.n_xlines, self.n_samples)
            self.dip = np.memmap(
                self.config.dip_file, dtype=np.float32,
                mode='r', shape=expected_shape
            )
            print(f"  Loaded dip: {expected_shape}")

    def _estimate_well_locations(self):
        """Estimate well inline/xline from lat/lon"""
        if not self.config.wells:
            return

        lats = [w.get('lat', 0) for w in self.config.wells.values()]
        lons = [w.get('lon', 0) for w in self.config.wells.values()]

        lat_min, lat_max = min(lats), max(lats)
        lon_min, lon_max = min(lons), max(lons)

        for well_name, well_data in self.config.wells.items():
            if lat_max > lat_min:
                norm_lat = (well_data.get('lat', 0) - lat_min) / (lat_max - lat_min)
            else:
                norm_lat = 0.5

            if lon_max > lon_min:
                norm_lon = (well_data.get('lon', 0) - lon_min) / (lon_max - lon_min)
            else:
                norm_lon = 0.5

            well_data['inline_est'] = int(norm_lat * (self.n_ilines - 1))
            well_data['xline_est'] = int(norm_lon * (self.n_xlines - 1))

        print(f"  Estimated locations for {len(self.config.wells)} wells")

    def _generate_seeds(self, formation_name: str) -> List[Tuple]:
        """Generate seed points for horizon tracking"""
        formation_data = self.config.formations[formation_name]
        typical_time = formation_data[0]

        seeds = []

        # Well-based seeds
        for well_name, well_data in self.config.wells.items():
            il = well_data.get('inline_est')
            xl = well_data.get('xline_est')
            if il is not None and xl is not None:
                seeds.append((il, xl, typical_time))

        # Grid-based seeds
        for i in range(0, self.n_ilines, self.config.grid_density):
            for j in range(0, self.n_xlines, self.config.grid_density):
                seeds.append((i, j, typical_time))

        print(f"  Generated {len(seeds)} seed points (grid={self.config.grid_density})")
        return seeds

    def _pick_event_locally(self, il_idx: int, xl_idx: int, time_idx: int,
                           window: int = 5, event_type: str = 'peak') -> int:
        """Pick specific seismic event around time index"""
        start = max(0, time_idx - window)
        end = min(self.n_samples, time_idx + window + 1)

        trace_segment = self.volume[il_idx, xl_idx, start:end]

        if event_type == 'peak':
            local_idx = np.argmax(trace_segment)
        elif event_type == 'trough':
            local_idx = np.argmin(trace_segment)
        else:
            local_idx = window

        return start + local_idx

    def _track_horizon(self, name: str, seed_points: List[Tuple],
                      horizon_type: str = 'peak') -> Tuple[np.ndarray, Dict]:
        """Track horizon using region growing"""

        print(f"  Tracking with parameters:")
        print(f"    Search window: ±{self.config.search_window} samples")
        print(f"    Similarity threshold: {self.config.similarity_threshold}")
        print(f"    Max dip: {self.config.max_dip}°")

        # Initialize
        horizon = np.full((self.n_ilines, self.n_xlines), np.nan)
        visited = np.zeros((self.n_ilines, self.n_xlines), dtype=bool)

        diagnostics = {
            'seeds_initialized': 0,
            'rejected_similarity': 0,
            'rejected_dip': 0,
            'rejected_coherence': 0,
            'accepted': 0
        }

        # Initialize seeds
        seed_queue = []
        for il, xl, time_ms in seed_points:
            try:
                il_idx = int(np.clip(il, 0, self.n_ilines - 1))
                xl_idx = int(np.clip(xl, 0, self.n_xlines - 1))
                time_idx = np.argmin(np.abs(self.time - time_ms))

                picked_idx = self._pick_event_locally(
                    il_idx, xl_idx, time_idx,
                    window=self.config.search_window,
                    event_type=horizon_type
                )

                horizon[il_idx, xl_idx] = self.time[picked_idx]
                visited[il_idx, xl_idx] = True
                seed_queue.append((il_idx, xl_idx, picked_idx))
                diagnostics['seeds_initialized'] += 1
            except:
                pass

        print(f"  Initialized {len(seed_queue)} seeds")

        # Define neighbors
        if self.config.use_8_connected:
            neighbors = [(0,1), (0,-1), (1,0), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]
        else:
            neighbors = [(0,1), (0,-1), (1,0), (-1,0)]

        # Region growing
        iteration = 0

        with tqdm(total=self.config.max_iterations, desc="Tracking") as pbar:
            while seed_queue and iteration < self.config.max_iterations:
                il_idx, xl_idx, time_idx = seed_queue.pop(0)
                ref_trace = self.volume[il_idx, xl_idx, :]

                for di, dj in neighbors:
                    ni = il_idx + di
                    nj = xl_idx + dj

                    # Bounds check
                    if not (0 <= ni < self.n_ilines and 0 <= nj < self.n_xlines):
                        continue

                    # Already visited
                    if visited[ni, nj]:
                        continue

                    # Coherence check
                    if self.config.fault_guided and self.coherence is not None:
                        coh = self.coherence[ni, nj, time_idx]
                        if coh < self.config.coherence_threshold:
                            diagnostics['rejected_coherence'] += 1
                            continue

                    # Get neighbor trace
                    neighbor_trace = self.volume[ni, nj, :]

                    # Search for best match
                    search_start = max(0, time_idx - self.config.search_window)
                    search_end = min(self.n_samples, time_idx + self.config.search_window + 1)

                    best_sim = -1
                    best_idx = time_idx

                    for test_idx in range(search_start, search_end):
                        win_size = self.config.correlation_window
                        win_start = max(0, test_idx - win_size)
                        win_end = min(self.n_samples, test_idx + win_size + 1)

                        ref_win = ref_trace[win_start:win_end]
                        test_win = neighbor_trace[win_start:win_end]

                        if len(ref_win) > 0 and len(test_win) > 0:
                            ref_norm = (ref_win - np.mean(ref_win)) / (np.std(ref_win) + 1e-6)
                            test_norm = (test_win - np.mean(test_win)) / (np.std(test_win) + 1e-6)

                            sim = np.corrcoef(ref_norm, test_norm)[0, 1]
                            if not np.isnan(sim) and sim > best_sim:
                                best_sim = sim
                                best_idx = test_idx

                    # Similarity check
                    if best_sim < self.config.similarity_threshold:
                        diagnostics['rejected_similarity'] += 1
                        continue

                    # Dip check
                    time_diff_ms = abs(self.time[best_idx] - self.time[time_idx])

                    if abs(di) + abs(dj) == 2:  # Diagonal
                        trace_spacing_m = self.config.trace_spacing * np.sqrt(2)
                    else:
                        trace_spacing_m = self.config.trace_spacing

                    if time_diff_ms > 0:
                        depth_diff_m = (time_diff_ms / 1000) * self.config.avg_velocity / 2
                        apparent_dip = np.degrees(np.arctan(depth_diff_m / trace_spacing_m))

                        if apparent_dip > self.config.max_dip:
                            diagnostics['rejected_dip'] += 1
                            continue

                    # Accept point
                    picked_idx = self._pick_event_locally(
                        ni, nj, best_idx, window=5, event_type=horizon_type
                    )

                    horizon[ni, nj] = self.time[picked_idx]
                    visited[ni, nj] = True
                    seed_queue.append((ni, nj, picked_idx))
                    diagnostics['accepted'] += 1

                iteration += 1
                pbar.update(1)

                if iteration % 100 == 0:
                    coverage = np.sum(visited) / visited.size * 100
                    pbar.set_description(f"Coverage: {coverage:.1f}%")

        # Post-processing
        if self.config.fill_small_gaps and SCIPY_AVAILABLE:
            horizon = self._fill_gaps(horizon)

        if self.config.smooth_sigma > 0 and SCIPY_AVAILABLE:
            horizon = self._smooth_horizon(horizon)

        # Statistics
        valid_points = ~np.isnan(horizon)
        coverage = np.sum(valid_points) / valid_points.size * 100

        stats = {
            'coverage': float(coverage),
            'iterations': iteration,
            'time_min': float(np.nanmin(horizon)),
            'time_max': float(np.nanmax(horizon)),
            'structural_relief': float(np.nanmax(horizon) - np.nanmin(horizon)),
            'diagnostics': diagnostics
        }

        print(f"  Coverage: {coverage:.1f}%")
        print(f"  Iterations: {iteration:,}")
        print(f"  Time range: {stats['time_min']:.0f} - {stats['time_max']:.0f} ms")

        self.diagnostics[name] = diagnostics

        return horizon, stats

    def _fill_gaps(self, horizon: np.ndarray) -> np.ndarray:
        """Fill small gaps in horizon"""
        mask = ~np.isnan(horizon)
        if not mask.any():
            return horizon

        coords = np.array(np.nonzero(mask)).T
        values = horizon[mask]

        y, x = np.mgrid[0:horizon.shape[0], 0:horizon.shape[1]]
        filled = griddata(coords, values, (y, x), method='linear')

        # Only fill small gaps
        gap_mask = np.isnan(horizon) & ~np.isnan(filled)
        gap_labels = measure.label(gap_mask) if SKIMAGE_AVAILABLE else np.zeros_like(gap_mask)

        if SKIMAGE_AVAILABLE:
            for region in measure.regionprops(gap_labels):
                if region.area <= self.config.max_gap_size:
                    horizon[gap_labels == region.label] = filled[gap_labels == region.label]

        return horizon

    def _smooth_horizon(self, horizon: np.ndarray) -> np.ndarray:
        """Smooth horizon surface"""
        mask = ~np.isnan(horizon)
        smoothed = horizon.copy()

        if mask.any():
            temp = horizon.copy()
            temp[~mask] = np.nanmean(horizon)
            smoothed_full = ndimage.gaussian_filter(temp, sigma=self.config.smooth_sigma)
            smoothed[mask] = smoothed_full[mask]

        return smoothed

    def _identify_closures(self, name: str, surface: np.ndarray) -> List[Dict]:
        """Identify structural closures"""
        if not SKIMAGE_AVAILABLE:
            return []

        valid_mask = ~np.isnan(surface)
        if not valid_mask.any():
            return []

        # Invert surface for watershed
        inverted = -surface.copy()
        inverted[~valid_mask] = np.nanmin(inverted) - 1000

        # Find local maxima
        local_max = peak_local_max(
            -inverted, min_distance=20,
            threshold_abs=self.config.min_closure_amplitude
        )

        if len(local_max) == 0:
            return []

        # Create markers
        markers = np.zeros_like(surface, dtype=int)
        markers[tuple(local_max.T)] = np.arange(1, len(local_max) + 1)

        # Watershed
        labels = watershed(inverted, markers, mask=valid_mask)

        # Analyze closures
        closures = []

        for region in measure.regionprops(labels):
            if region.area < self.config.min_closure_area:
                continue

            mask = labels == region.label
            closure_values = surface[mask]

            if len(closure_values) == 0:
                continue

            amplitude = np.nanmax(closure_values) - np.nanmin(closure_values)

            if amplitude >= self.config.min_closure_amplitude:
                closures.append({
                    'id': len(closures) + 1,
                    'area_traces': int(region.area),
                    'centroid_il': float(region.centroid[0]),
                    'centroid_xl': float(region.centroid[1]),
                    'amplitude_ms': float(amplitude),
                    'max_time': float(np.nanmax(closure_values)),
                    'min_time': float(np.nanmin(closure_values))
                })

        closures.sort(key=lambda x: x['amplitude_ms'], reverse=True)

        print(f"  Identified {len(closures)} structural closures")

        return closures

    def _visualize_horizon(self, formation_name: str):
        """Create horizon visualization"""
        if formation_name not in self.horizons:
            return

        horizon_data = self.horizons[formation_name]
        surface = horizon_data['surface']

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'{formation_name} - Horizon Interpretation', fontsize=14, fontweight='bold')

        valid_mask = ~np.isnan(surface)
        if valid_mask.any():
            vmin, vmax = np.nanpercentile(surface, [2, 98])

            # Time structure
            im1 = axes[0].imshow(surface, cmap='rainbow', aspect='auto', vmin=vmin, vmax=vmax)
            axes[0].set_title('Time Structure (TWT ms)')
            axes[0].set_xlabel('Crossline')
            axes[0].set_ylabel('Inline')
            plt.colorbar(im1, ax=axes[0], label='TWT (ms)')

            # Coverage
            axes[1].imshow(valid_mask, cmap='RdYlGn', aspect='auto')
            axes[1].set_title(f'Coverage: {horizon_data["stats"]["coverage"]:.1f}%')
            axes[1].set_xlabel('Crossline')
            axes[1].set_ylabel('Inline')

            # Gradient
            grad = np.gradient(surface)
            grad_mag = np.sqrt(grad[0]**2 + grad[1]**2)
            im3 = axes[2].imshow(grad_mag, cmap='hot', aspect='auto')
            axes[2].set_title('Structural Gradient')
            axes[2].set_xlabel('Crossline')
            axes[2].set_ylabel('Inline')
            plt.colorbar(im3, ax=axes[2], label='Gradient (ms/trace)')

        plt.tight_layout()

        filename = self.output_dir / "figures" / f"{formation_name}_interpretation.png"
        plt.savefig(filename, dpi=self.config.figure_dpi, bbox_inches='tight')
        plt.close()

        print(f"  Saved: {filename}")

    def _visualize_structure_map(self, formation_name: str):
        """Create structure map with closures"""
        if formation_name not in self.horizons:
            return

        horizon_data = self.horizons[formation_name]
        surface = horizon_data['surface']

        fig, ax = plt.subplots(figsize=(12, 10))

        valid_mask = ~np.isnan(surface)
        if valid_mask.any():
            vmin, vmax = np.nanpercentile(surface, [2, 98])

            im = ax.imshow(surface, cmap='terrain', aspect='auto', vmin=vmin, vmax=vmax)

            # Contours
            contours = ax.contour(
                surface,
                levels=np.arange(vmin, vmax, self.config.contour_interval),
                colors='black', linewidths=0.5, alpha=0.7
            )
            ax.clabel(contours, inline=True, fontsize=8, fmt='%d ms')

            plt.colorbar(im, ax=ax, label='TWT (ms)')
            ax.set_title(f'{formation_name} - Structure Map', fontsize=12, fontweight='bold')
            ax.set_xlabel('Crossline')
            ax.set_ylabel('Inline')

        plt.tight_layout()

        filename = self.output_dir / "figures" / f"{formation_name}_structure_map.png"
        plt.savefig(filename, dpi=self.config.figure_dpi, bbox_inches='tight')
        plt.close()

        print(f"  Saved: {filename}")


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Horizon Interpretation Automation v5.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python horizon_interpretation_automation.py input.segy -o outputs
  python horizon_interpretation_automation.py -c config.json
  python horizon_interpretation_automation.py input.segy --grid-density 25 --similarity 0.35
        """
    )

    parser.add_argument("seismic_file", nargs="?", help="Input SEG-Y file")
    parser.add_argument("-c", "--config", help="Configuration JSON file")
    parser.add_argument("-o", "--output-dir", default="horizon_outputs", help="Output directory")
    parser.add_argument("--coherence", help="Coherence volume file")
    parser.add_argument("--dip", help="Dip volume file")
    parser.add_argument("--grid-density", type=int, default=50, help="Grid seed spacing")
    parser.add_argument("--search-window", type=int, default=30, help="Search window in samples")
    parser.add_argument("--similarity", type=float, default=0.4, help="Similarity threshold")
    parser.add_argument("--max-dip", type=float, default=85, help="Maximum dip in degrees")
    parser.add_argument("--max-iterations", type=int, default=200000, help="Maximum iterations")
    parser.add_argument("--no-figures", action="store_true", help="Disable figure generation")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    parser.add_argument("--webhook", help="Webhook URL for progress updates")
    parser.add_argument("--create-config", help="Create default config file and exit")

    args = parser.parse_args()

    # Create default config
    if args.create_config:
        config = HorizonInterpretationConfig()
        config.to_json(args.create_config)
        print(f"Created config file: {args.create_config}")
        return

    # Load or create config
    if args.config:
        config = HorizonInterpretationConfig.from_json(args.config)
    else:
        if not args.seismic_file:
            parser.error("Either seismic_file or --config is required")

        config = HorizonInterpretationConfig(
            seismic_file=args.seismic_file,
            output_dir=args.output_dir,
            coherence_file=args.coherence,
            dip_file=args.dip,
            grid_density=args.grid_density,
            search_window=args.search_window,
            similarity_threshold=args.similarity,
            max_dip=args.max_dip,
            max_iterations=args.max_iterations,
            save_figures=not args.no_figures,
            enable_cache=not args.no_cache,
            webhook_url=args.webhook
        )

    # Run automation
    automation = HorizonInterpretationAutomation(config)
    results = automation.run()

    # Return exit code
    return 0 if results.success else 1


if __name__ == "__main__":
    exit(main())
