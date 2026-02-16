"""
Interactive Seismic Viewer with AI Interpretation
==================================================
PhD-grade seismic visualization for Bornu Chad Basin.

Features:
- 3D volume navigation (inline, xline, time slices)
- 2D line display
- Well-to-seismic tie validation with correlation metrics
- AI interpretation via Ollama (Llava for images, Qwen for text)
- Time and Depth domain display

Author: Moses Ekene Obasi
PhD Research - University of Calabar
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import json
import io
import base64
import tempfile
import subprocess
from scipy import signal, interpolate
from scipy.ndimage import gaussian_filter
import threading

# Try to import segyio
try:
    import segyio
    SEGYIO_AVAILABLE = True
except ImportError:
    SEGYIO_AVAILABLE = False
    print("Warning: segyio not available. Install with: pip install segyio")

# Try GPU acceleration
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

# Custom seismic colormap (red-white-blue)
SEISMIC_CMAP = LinearSegmentedColormap.from_list(
    'seismic_custom',
    [(0, 'blue'), (0.25, 'lightblue'), (0.5, 'white'), (0.75, 'lightyellow'), (1, 'red')]
)

# Alternative gray colormap
GRAY_CMAP = plt.cm.gray


class SeismicDataLoader:
    """Handles loading of 3D and 2D seismic data from SEGY files."""

    def __init__(self):
        self.segy_3d = None
        self.segy_3d_path = None
        self.segy_2d_files = {}
        self.volume_cache = {}

        # Survey geometry (will be populated from SEGY)
        self.il_min = self.il_max = 0
        self.xl_min = self.xl_max = 0
        self.n_samples = 0
        self.sample_rate = 4.0  # ms
        self.record_length = 0  # ms

        # Velocity model for depth conversion
        self.v0 = 1800  # m/s at surface
        self.k = 0.5    # velocity gradient s^-1

    def load_3d_volume(self, segy_path: str) -> bool:
        """Load 3D SEGY volume and extract geometry."""
        if not SEGYIO_AVAILABLE:
            return False

        try:
            self.segy_3d_path = segy_path
            self.segy_3d = segyio.open(segy_path, 'r', strict=False)

            # Get geometry
            self.il_min = int(np.min(self.segy_3d.ilines))
            self.il_max = int(np.max(self.segy_3d.ilines))
            self.xl_min = int(np.min(self.segy_3d.xlines))
            self.xl_max = int(np.max(self.segy_3d.xlines))
            self.n_samples = self.segy_3d.samples.size
            self.sample_rate = segyio.tools.dt(self.segy_3d) / 1000.0  # Convert to ms
            self.record_length = self.n_samples * self.sample_rate

            print(f"Loaded 3D volume: IL {self.il_min}-{self.il_max}, XL {self.xl_min}-{self.xl_max}")
            print(f"Samples: {self.n_samples}, Sample rate: {self.sample_rate}ms, Record: {self.record_length}ms")

            return True
        except Exception as e:
            print(f"Error loading 3D SEGY: {e}")
            return False

    def get_inline(self, il: int) -> np.ndarray:
        """Extract a single inline section."""
        if self.segy_3d is None:
            return None

        cache_key = f"il_{il}"
        if cache_key in self.volume_cache:
            return self.volume_cache[cache_key]

        try:
            data = self.segy_3d.iline[il]
            # Limit cache size
            if len(self.volume_cache) > 20:
                self.volume_cache.pop(next(iter(self.volume_cache)))
            self.volume_cache[cache_key] = data
            return data
        except:
            return None

    def get_xline(self, xl: int) -> np.ndarray:
        """Extract a single crossline section."""
        if self.segy_3d is None:
            return None

        cache_key = f"xl_{xl}"
        if cache_key in self.volume_cache:
            return self.volume_cache[cache_key]

        try:
            data = self.segy_3d.xline[xl]
            if len(self.volume_cache) > 20:
                self.volume_cache.pop(next(iter(self.volume_cache)))
            self.volume_cache[cache_key] = data
            return data
        except:
            return None

    def get_time_slice(self, time_ms: float) -> np.ndarray:
        """Extract a time slice at given TWT."""
        if self.segy_3d is None:
            return None

        sample_idx = int(time_ms / self.sample_rate)
        if sample_idx < 0 or sample_idx >= self.n_samples:
            return None

        cache_key = f"ts_{time_ms}"
        if cache_key in self.volume_cache:
            return self.volume_cache[cache_key]

        try:
            # Build time slice by reading all inlines
            n_il = self.il_max - self.il_min + 1
            n_xl = self.xl_max - self.xl_min + 1

            time_slice = np.zeros((n_il, n_xl))
            for i, il in enumerate(range(self.il_min, self.il_max + 1, max(1, n_il // 100))):
                try:
                    inline_data = self.segy_3d.iline[il]
                    time_slice[i, :] = inline_data[:, sample_idx] if inline_data.shape[1] > sample_idx else 0
                except:
                    pass

            if len(self.volume_cache) > 20:
                self.volume_cache.pop(next(iter(self.volume_cache)))
            self.volume_cache[cache_key] = time_slice
            return time_slice
        except Exception as e:
            print(f"Error extracting time slice: {e}")
            return None

    def load_2d_line(self, segy_path: str) -> Tuple[np.ndarray, dict]:
        """Load a 2D seismic line."""
        if not SEGYIO_AVAILABLE:
            return None, {}

        try:
            with segyio.open(segy_path, 'r', ignore_geometry=True) as f:
                data = segyio.tools.collect(f.trace[:])
                n_traces = len(f.trace)
                n_samples = f.samples.size
                sample_rate = segyio.tools.dt(f) / 1000.0

                info = {
                    'n_traces': n_traces,
                    'n_samples': n_samples,
                    'sample_rate': sample_rate,
                    'record_length': n_samples * sample_rate,
                    'filename': Path(segy_path).name
                }
                return data.T, info  # Transpose to (samples, traces)
        except Exception as e:
            print(f"Error loading 2D line: {e}")
            return None, {}

    def time_to_depth(self, twt_ms: float) -> float:
        """Convert TWT (ms) to depth (m) using velocity model."""
        # V(z) = V0 + k*z
        # t = 2 * integral(dz/V) = (2/k) * ln((V0 + k*z)/V0)
        # Solving for z: z = (V0/k) * (exp(k*t/2) - 1)
        twt_s = twt_ms / 1000.0
        depth = (self.v0 / self.k) * (np.exp(self.k * twt_s / 2) - 1)
        return depth

    def depth_to_time(self, depth_m: float) -> float:
        """Convert depth (m) to TWT (ms) using velocity model."""
        twt_s = (2 / self.k) * np.log((self.v0 + self.k * depth_m) / self.v0)
        return twt_s * 1000.0

    def close(self):
        """Close open SEGY files."""
        if self.segy_3d is not None:
            self.segy_3d.close()
            self.segy_3d = None


class WellTieValidator:
    """
    Well-to-Seismic Tie Validation with correlation metrics.

    This is THE MOST IMPORTANT step in seismic interpretation.
    Without a good well tie, your horizons are meaningless.
    """

    def __init__(self, seismic_loader: SeismicDataLoader):
        self.loader = seismic_loader
        self.wells = {}
        self.ties = {}

    def load_well_logs(self, well_name: str, csv_path: str) -> dict:
        """Load well log data from CSV."""
        import pandas as pd

        try:
            df = pd.read_csv(csv_path)

            # Standardize column names (handle various naming conventions)
            col_map = {}
            for col in df.columns:
                col_lower = col.lower()
                # Depth columns
                if 'depth' in col_lower or col_lower == 'dept':
                    col_map[col] = 'Depth_m'
                # Sonic columns (DT) - various formats
                elif col_lower in ['dt', 'dt8', 'dtco', 'sonic', 'dt_usm', 'dt_usft']:
                    col_map[col] = 'DT'
                # Density columns (RHOB)
                elif col_lower in ['rhob', 'rhob_gcc', 'den', 'density', 'rho', 'bulk_density']:
                    col_map[col] = 'RHOB'
                # Gamma ray
                elif col_lower in ['gr', 'gr_api', 'gamma', 'gamma_ray']:
                    col_map[col] = 'GR'
                # Velocity (pre-computed)
                elif 'velocity' in col_lower or col_lower == 'vp':
                    col_map[col] = 'Velocity'
                # Porosity
                elif 'phi' in col_lower or col_lower == 'porosity':
                    col_map[col] = 'Porosity'
                # Water saturation
                elif col_lower in ['sw', 'water_sat']:
                    col_map[col] = 'Sw'
                # Shale volume
                elif col_lower in ['vsh', 'vcl', 'shale_vol']:
                    col_map[col] = 'Vsh'

            df = df.rename(columns=col_map)
            self.wells[well_name] = df

            # Log what columns we found
            available_cols = [c for c in ['Depth_m', 'DT', 'RHOB', 'Velocity', 'GR', 'Porosity', 'Sw'] if c in df.columns]

            return {'status': 'success', 'samples': len(df), 'columns': available_cols}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    def calculate_synthetic(self, well_name: str, wavelet_freq: float = 12.0) -> dict:
        """
        Generate synthetic seismogram with full validation metrics.

        Returns:
        - synthetic trace
        - reflection coefficients
        - acoustic impedance
        - time-depth curve
        - correlation metrics
        """
        if well_name not in self.wells:
            return {'status': 'error', 'message': f'Well {well_name} not loaded'}

        df = self.wells[well_name].copy()

        # Get required logs
        if 'Depth_m' not in df.columns:
            return {'status': 'error', 'message': 'Depth column not found'}

        depth = df['Depth_m'].values

        # Get velocity - prefer pre-computed, otherwise calculate from sonic
        if 'Velocity' in df.columns:
            velocity = df['Velocity'].values
            velocity = np.where((velocity <= 0) | (velocity > 8000) | np.isnan(velocity), np.nan, velocity)
        elif 'DT' in df.columns:
            dt = df['DT'].values
            # Check if DT is in us/ft or us/m
            median_dt = np.nanmedian(dt)
            if median_dt > 100:  # Likely us/m
                dt = np.where((dt <= 0) | (dt > 1000) | np.isnan(dt), np.nan, dt)
                velocity = 1e6 / dt  # us/m to m/s
            else:  # us/ft
                dt = np.where((dt <= 0) | (dt > 300) | np.isnan(dt), np.nan, dt)
                velocity = 304800.0 / dt  # Convert us/ft to m/s
        else:
            return {'status': 'error', 'message': 'Sonic (DT) or Velocity log not found'}

        # Get density
        if 'RHOB' in df.columns:
            rhob = df['RHOB'].values
            rhob = np.where((rhob <= 0) | (rhob > 3.5) | np.isnan(rhob), np.nan, rhob)
            density = rhob * 1000  # g/cc to kg/m3
        else:
            # Estimate from velocity using Gardner's relation
            density = 310 * (velocity ** 0.25)

        # Calculate Acoustic Impedance
        ai = velocity * density

        # Calculate Reflection Coefficients
        rc = np.zeros_like(ai)
        for i in range(1, len(ai)):
            if not (np.isnan(ai[i]) or np.isnan(ai[i-1])):
                rc[i] = (ai[i] - ai[i-1]) / (ai[i] + ai[i-1])
        rc = np.clip(rc, -0.5, 0.5)

        # Create time-depth relationship
        # TWT = 2 * integral(dz / V)
        valid_mask = ~np.isnan(velocity) & (velocity > 0)
        if np.sum(valid_mask) < 10:
            return {'status': 'error', 'message': 'Insufficient valid velocity data'}

        depth_valid = depth[valid_mask]
        vel_valid = velocity[valid_mask]

        # Calculate TWT
        dz = np.diff(depth_valid)
        v_avg = (vel_valid[:-1] + vel_valid[1:]) / 2
        dt_owt = dz / v_avg  # One-way time in seconds
        twt = np.zeros(len(depth_valid))
        twt[1:] = 2 * np.cumsum(dt_owt)  # Two-way time in seconds
        twt_ms = twt * 1000  # Convert to ms

        # Create Ricker wavelet
        sample_rate = 0.004  # 4ms sampling
        duration = 0.128  # 128ms wavelet
        t_wav = np.arange(-duration/2, duration/2, sample_rate)
        pi2 = (np.pi * wavelet_freq * t_wav) ** 2
        wavelet = (1 - 2 * pi2) * np.exp(-pi2)
        wavelet = wavelet / np.max(np.abs(wavelet))

        # Resample RC to regular time sampling
        twt_regular = np.arange(0, twt_ms[-1], sample_rate * 1000)
        rc_interp = np.interp(twt_regular, twt_ms, rc[valid_mask])

        # Convolve RC with wavelet to create synthetic
        synthetic = np.convolve(rc_interp, wavelet, mode='same')

        # Store results
        self.ties[well_name] = {
            'depth': depth_valid,
            'twt_ms': twt_ms,
            'velocity': vel_valid,
            'density': density[valid_mask] if 'RHOB' in df.columns else density[valid_mask],
            'ai': ai[valid_mask],
            'rc': rc[valid_mask],
            'synthetic': synthetic,
            'twt_regular': twt_regular,
            'wavelet': wavelet,
            'wavelet_freq': wavelet_freq,
            'sample_rate': sample_rate * 1000
        }

        return {
            'status': 'success',
            'depth_range': [float(depth_valid[0]), float(depth_valid[-1])],
            'twt_range': [float(twt_ms[0]), float(twt_ms[-1])],
            'ai_range': [float(np.nanmin(ai[valid_mask])), float(np.nanmax(ai[valid_mask]))],
            'rc_range': [float(np.nanmin(rc[valid_mask])), float(np.nanmax(rc[valid_mask]))]
        }

    def extract_seismic_trace(self, well_name: str, il: int, xl: int) -> dict:
        """Extract seismic trace at well location for correlation."""
        if well_name not in self.ties:
            return {'status': 'error', 'message': 'Calculate synthetic first'}

        if self.loader.segy_3d is None:
            return {'status': 'error', 'message': 'Load 3D seismic first'}

        try:
            # Get inline data
            inline_data = self.loader.get_inline(il)
            if inline_data is None:
                return {'status': 'error', 'message': f'Could not extract inline {il}'}

            # Find crossline index
            xl_idx = xl - self.loader.xl_min
            if xl_idx < 0 or xl_idx >= inline_data.shape[0]:
                return {'status': 'error', 'message': f'Crossline {xl} out of range'}

            # Extract trace
            seismic_trace = inline_data[xl_idx, :]

            # Create time axis
            time_axis = np.arange(len(seismic_trace)) * self.loader.sample_rate

            # Store
            self.ties[well_name]['seismic_trace'] = seismic_trace
            self.ties[well_name]['seismic_time'] = time_axis
            self.ties[well_name]['well_il'] = il
            self.ties[well_name]['well_xl'] = xl

            return {'status': 'success', 'n_samples': len(seismic_trace)}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    def correlate_well_tie(self, well_name: str, max_shift_ms: float = 100) -> dict:
        """
        Correlate synthetic with seismic trace to find optimal tie.

        Returns:
        - Correlation coefficient
        - Optimal time shift
        - Phase rotation if needed
        """
        if well_name not in self.ties:
            return {'status': 'error', 'message': 'Calculate synthetic first'}

        tie = self.ties[well_name]

        if 'seismic_trace' not in tie:
            return {'status': 'error', 'message': 'Extract seismic trace first'}

        synthetic = tie['synthetic']
        seismic_trace = tie['seismic_trace']
        twt_regular = tie['twt_regular']
        seismic_time = tie['seismic_time']

        # Resample seismic to match synthetic time range
        twt_min, twt_max = twt_regular[0], twt_regular[-1]

        # Find overlapping window
        seismic_mask = (seismic_time >= twt_min) & (seismic_time <= twt_max)
        if np.sum(seismic_mask) < 50:
            return {'status': 'error', 'message': 'Insufficient overlap between synthetic and seismic'}

        # Interpolate seismic to synthetic time sampling
        seismic_interp = np.interp(twt_regular, seismic_time, seismic_trace)

        # Normalize both traces
        syn_norm = (synthetic - np.mean(synthetic)) / (np.std(synthetic) + 1e-10)
        seis_norm = (seismic_interp - np.mean(seismic_interp)) / (np.std(seismic_interp) + 1e-10)

        # Cross-correlation to find optimal shift
        max_shift_samples = int(max_shift_ms / tie['sample_rate'])
        correlation = signal.correlate(seis_norm, syn_norm, mode='full')
        lags = signal.correlation_lags(len(seis_norm), len(syn_norm), mode='full')

        # Find peak within allowed shift range
        valid_range = np.abs(lags) <= max_shift_samples
        corr_valid = correlation.copy()
        corr_valid[~valid_range] = -np.inf

        peak_idx = np.argmax(corr_valid)
        optimal_shift_samples = lags[peak_idx]
        optimal_shift_ms = optimal_shift_samples * tie['sample_rate']

        # Calculate correlation at optimal shift
        if optimal_shift_samples > 0:
            syn_shifted = np.roll(synthetic, optimal_shift_samples)
        else:
            syn_shifted = np.roll(synthetic, optimal_shift_samples)

        syn_shifted_norm = (syn_shifted - np.mean(syn_shifted)) / (np.std(syn_shifted) + 1e-10)
        corr_coef = np.corrcoef(syn_shifted_norm, seis_norm)[0, 1]

        # Try different phase rotations to improve tie
        best_phase = 0
        best_corr = corr_coef

        for phase in range(0, 180, 10):
            phase_rad = np.radians(phase)
            analytic = signal.hilbert(synthetic)
            rotated = np.real(analytic) * np.cos(phase_rad) + np.imag(analytic) * np.sin(phase_rad)
            rotated_norm = (rotated - np.mean(rotated)) / (np.std(rotated) + 1e-10)

            if optimal_shift_samples != 0:
                rotated_norm = np.roll(rotated_norm, optimal_shift_samples)

            phase_corr = np.corrcoef(rotated_norm, seis_norm)[0, 1]
            if phase_corr > best_corr:
                best_corr = phase_corr
                best_phase = phase

        # Store correlation results
        tie['correlation'] = {
            'coefficient': float(best_corr),
            'optimal_shift_ms': float(optimal_shift_ms),
            'optimal_phase_deg': float(best_phase),
            'seismic_interp': seismic_interp,
            'synthetic_shifted': syn_shifted if optimal_shift_samples == 0 else np.roll(synthetic, optimal_shift_samples)
        }

        # Quality assessment
        if best_corr >= 0.7:
            quality = "EXCELLENT"
        elif best_corr >= 0.5:
            quality = "GOOD"
        elif best_corr >= 0.3:
            quality = "FAIR"
        else:
            quality = "POOR"

        return {
            'status': 'success',
            'correlation_coefficient': float(best_corr),
            'optimal_shift_ms': float(optimal_shift_ms),
            'optimal_phase_deg': float(best_phase),
            'quality': quality,
            'interpretation': f"Well tie quality is {quality}. Correlation r={best_corr:.3f}, shift={optimal_shift_ms:.1f}ms, phase={best_phase}deg"
        }


class SeismicViewerGUI:
    """
    Interactive Seismic Viewer GUI.

    Features:
    - 3D volume navigation (inline, xline, time slice)
    - 2D line display
    - Time/Depth domain toggle
    - AI interpretation integration
    - Well tie display
    """

    def __init__(self, parent_frame, ollama_client=None, colors=None):
        self.parent = parent_frame
        self.ollama = ollama_client
        self.colors = colors or {
            'bg': '#1a1a2e',
            'surface': '#16213e',
            'overlay': '#0f3460',
            'fg': '#e6e6e6',
            'accent': '#e94560',
            'success': '#4ecca3',
            'warning': '#ffc107'
        }

        # Data
        self.loader = SeismicDataLoader()
        self.well_tie = WellTieValidator(self.loader)
        self.current_view = 'inline'  # inline, xline, timeslice, 2d
        self.domain = 'time'  # time or depth

        # Current display parameters
        self.current_il = 5500
        self.current_xl = 5900
        self.current_time = 1500

        # Figure for display
        self.fig = None
        self.canvas = None
        self.ax = None
        self.colorbar = None  # Track colorbar to remove it on redraw

        # Fault data for overlay
        self.faults = []
        self.show_faults = tk.BooleanVar(value=False)
        self._load_fault_data()

        self._create_gui()

    def _load_fault_data(self):
        """Load fault data from interpretation results if available."""
        try:
            results_path = Path(__file__).parent / 'interpretation' / 'real_outputs' / 'interpretation_results.json'
            if results_path.exists():
                with open(results_path, 'r') as f:
                    results = json.load(f)
                    if 'faults' in results:
                        self.faults = results['faults']
                        print(f"Loaded {len(self.faults)} faults for overlay")
        except Exception as e:
            print(f"Could not load fault data: {e}")
            self.faults = []

    def _create_gui(self):
        """Create the viewer GUI."""
        # Main container
        main_frame = tk.Frame(self.parent, bg=self.colors['bg'])
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Control panel (left)
        control_frame = tk.Frame(main_frame, bg=self.colors['surface'], width=300)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        control_frame.pack_propagate(False)

        # Display panel (right)
        display_frame = tk.Frame(main_frame, bg=self.colors['bg'])
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self._create_controls(control_frame)
        self._create_display(display_frame)

    def _create_controls(self, parent):
        """Create control panel."""
        # Title
        tk.Label(
            parent,
            text="Seismic Viewer",
            font=('Segoe UI', 14, 'bold'),
            bg=self.colors['surface'],
            fg=self.colors['accent']
        ).pack(pady=10)

        # Load Data Section
        load_frame = tk.LabelFrame(parent, text="Load Data", bg=self.colors['surface'], fg=self.colors['fg'])
        load_frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Button(
            load_frame,
            text="Load 3D Volume",
            command=self._load_3d_volume,
            bg=self.colors['accent'],
            fg='white'
        ).pack(fill=tk.X, padx=5, pady=2)

        tk.Button(
            load_frame,
            text="Load 2D Line",
            command=self._load_2d_line,
            bg=self.colors['overlay'],
            fg='white'
        ).pack(fill=tk.X, padx=5, pady=2)

        # Status label
        self.status_label = tk.Label(
            load_frame,
            text="No data loaded",
            font=('Segoe UI', 9),
            bg=self.colors['surface'],
            fg=self.colors['warning'],
            wraplength=250
        )
        self.status_label.pack(pady=5)

        # View Selection
        view_frame = tk.LabelFrame(parent, text="View Mode", bg=self.colors['surface'], fg=self.colors['fg'])
        view_frame.pack(fill=tk.X, padx=10, pady=5)

        self.view_var = tk.StringVar(value='inline')
        for view, label in [('inline', 'Inline Section'), ('xline', 'Crossline Section'),
                           ('timeslice', 'Time Slice'), ('2d', '2D Line')]:
            tk.Radiobutton(
                view_frame,
                text=label,
                variable=self.view_var,
                value=view,
                command=self._on_view_change,
                bg=self.colors['surface'],
                fg=self.colors['fg'],
                selectcolor=self.colors['overlay']
            ).pack(anchor=tk.W, padx=5)

        # Navigation
        nav_frame = tk.LabelFrame(parent, text="Navigation", bg=self.colors['surface'], fg=self.colors['fg'])
        nav_frame.pack(fill=tk.X, padx=10, pady=5)

        # Inline slider
        tk.Label(nav_frame, text="Inline:", bg=self.colors['surface'], fg=self.colors['fg']).pack(anchor=tk.W, padx=5)
        self.il_slider = tk.Scale(
            nav_frame,
            from_=5047, to=6047,
            orient=tk.HORIZONTAL,
            bg=self.colors['surface'],
            fg=self.colors['fg'],
            highlightthickness=0,
            command=self._on_il_change
        )
        self.il_slider.set(5500)
        self.il_slider.pack(fill=tk.X, padx=5)

        # Crossline slider
        tk.Label(nav_frame, text="Crossline:", bg=self.colors['surface'], fg=self.colors['fg']).pack(anchor=tk.W, padx=5)
        self.xl_slider = tk.Scale(
            nav_frame,
            from_=4885, to=7020,
            orient=tk.HORIZONTAL,
            bg=self.colors['surface'],
            fg=self.colors['fg'],
            highlightthickness=0,
            command=self._on_xl_change
        )
        self.xl_slider.set(5900)
        self.xl_slider.pack(fill=tk.X, padx=5)

        # Time slider
        tk.Label(nav_frame, text="Time (ms):", bg=self.colors['surface'], fg=self.colors['fg']).pack(anchor=tk.W, padx=5)
        self.time_slider = tk.Scale(
            nav_frame,
            from_=0, to=8000,
            orient=tk.HORIZONTAL,
            bg=self.colors['surface'],
            fg=self.colors['fg'],
            highlightthickness=0,
            command=self._on_time_change
        )
        self.time_slider.set(1500)
        self.time_slider.pack(fill=tk.X, padx=5)

        # Domain toggle
        domain_frame = tk.LabelFrame(parent, text="Domain", bg=self.colors['surface'], fg=self.colors['fg'])
        domain_frame.pack(fill=tk.X, padx=10, pady=5)

        self.domain_var = tk.StringVar(value='time')
        tk.Radiobutton(
            domain_frame,
            text="Time (TWT)",
            variable=self.domain_var,
            value='time',
            command=self._on_domain_change,
            bg=self.colors['surface'],
            fg=self.colors['fg'],
            selectcolor=self.colors['overlay']
        ).pack(side=tk.LEFT, padx=10)

        tk.Radiobutton(
            domain_frame,
            text="Depth",
            variable=self.domain_var,
            value='depth',
            command=self._on_domain_change,
            bg=self.colors['surface'],
            fg=self.colors['fg'],
            selectcolor=self.colors['overlay']
        ).pack(side=tk.LEFT, padx=10)

        # Fault overlay checkbox
        fault_frame = tk.Frame(nav_frame, bg=self.colors['surface'])
        fault_frame.pack(fill=tk.X, padx=5, pady=5)

        tk.Checkbutton(
            fault_frame,
            text="Show Fault Lines",
            variable=self.show_faults,
            command=self._display_section,
            bg=self.colors['surface'],
            fg=self.colors['fg'],
            selectcolor=self.colors['overlay'],
            activebackground=self.colors['surface'],
            activeforeground=self.colors['fg']
        ).pack(side=tk.LEFT)

        self.fault_count_label = tk.Label(
            fault_frame,
            text=f"({len(self.faults)} faults loaded)" if self.faults else "(no faults)",
            font=('Segoe UI', 8),
            bg=self.colors['surface'],
            fg=self.colors['warning'] if self.faults else self.colors['fg']
        )
        self.fault_count_label.pack(side=tk.LEFT, padx=5)

        # Display button
        tk.Button(
            nav_frame,
            text="Display Section",
            command=self._display_section,
            bg=self.colors['success'],
            fg='white',
            font=('Segoe UI', 10, 'bold')
        ).pack(fill=tk.X, padx=5, pady=10)

        # AI Interpretation
        ai_frame = tk.LabelFrame(parent, text="AI Interpretation", bg=self.colors['surface'], fg=self.colors['fg'])
        ai_frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Button(
            ai_frame,
            text="Interpret Current View",
            command=self._ai_interpret,
            bg=self.colors['accent'],
            fg='white'
        ).pack(fill=tk.X, padx=5, pady=5)

        self.ai_response = tk.Text(
            ai_frame,
            height=8,
            bg=self.colors['overlay'],
            fg=self.colors['fg'],
            wrap=tk.WORD,
            font=('Consolas', 9)
        )
        self.ai_response.pack(fill=tk.X, padx=5, pady=5)

    def _create_display(self, parent):
        """Create display panel with matplotlib figure."""
        # Create figure
        self.fig = Figure(figsize=(10, 8), facecolor=self.colors['bg'])
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor(self.colors['surface'])

        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.draw()

        # Toolbar
        toolbar_frame = tk.Frame(parent, bg=self.colors['bg'])
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        toolbar.update()

        # Canvas widget
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Show placeholder
        self._show_placeholder()

    def _show_placeholder(self):
        """Show placeholder when no data loaded."""
        # Clear figure properly
        self._clear_figure()

        self.ax.text(
            0.5, 0.5,
            "Load seismic data to begin\n\n"
            "Click 'Load 3D Volume' or 'Load 2D Line'\n\n"
            "Then use the navigation controls to browse",
            ha='center', va='center',
            fontsize=14,
            color=self.colors['fg'],
            transform=self.ax.transAxes
        )
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.canvas.draw()

    def _load_3d_volume(self):
        """Load 3D SEGY volume."""
        filepath = filedialog.askopenfilename(
            title="Select 3D SEGY File",
            filetypes=[("SEGY files", "*.segy *.sgy *.SEGY *.SGY"), ("All files", "*")]
        )

        if filepath:
            self.status_label.config(text="Loading 3D volume...", fg=self.colors['warning'])
            self.parent.update()

            if self.loader.load_3d_volume(filepath):
                # Update slider ranges
                self.il_slider.config(from_=self.loader.il_min, to=self.loader.il_max)
                self.xl_slider.config(from_=self.loader.xl_min, to=self.loader.xl_max)
                self.time_slider.config(from_=0, to=self.loader.record_length)

                # Set initial positions
                self.il_slider.set((self.loader.il_min + self.loader.il_max) // 2)
                self.xl_slider.set((self.loader.xl_min + self.loader.xl_max) // 2)
                self.time_slider.set(self.loader.record_length // 4)

                self.status_label.config(
                    text=f"Loaded: IL {self.loader.il_min}-{self.loader.il_max}, "
                         f"XL {self.loader.xl_min}-{self.loader.xl_max}",
                    fg=self.colors['success']
                )

                # Display initial section
                self._display_section()
            else:
                self.status_label.config(text="Failed to load volume", fg=self.colors['accent'])

    def _load_2d_line(self):
        """Load 2D SEGY line."""
        filepath = filedialog.askopenfilename(
            title="Select 2D SEGY File",
            filetypes=[("SEGY files", "*.segy *.sgy *.SEGY *.SGY"), ("All files", "*")]
        )

        if filepath:
            self.status_label.config(text="Loading 2D line...", fg=self.colors['warning'])
            self.parent.update()

            data, info = self.loader.load_2d_line(filepath)
            if data is not None:
                self.loader.segy_2d_files[info['filename']] = {'data': data, 'info': info}
                self.status_label.config(
                    text=f"Loaded 2D: {info['n_traces']} traces, {info['record_length']:.0f}ms",
                    fg=self.colors['success']
                )
                self.view_var.set('2d')
                self._display_2d_line(data, info)
            else:
                self.status_label.config(text="Failed to load 2D line", fg=self.colors['accent'])

    def _on_view_change(self):
        """Handle view mode change."""
        self.current_view = self.view_var.get()
        self._display_section()

    def _on_il_change(self, val):
        """Handle inline slider change."""
        self.current_il = int(float(val))

    def _on_xl_change(self, val):
        """Handle crossline slider change."""
        self.current_xl = int(float(val))

    def _on_time_change(self, val):
        """Handle time slider change."""
        self.current_time = float(val)

    def _on_domain_change(self):
        """Handle domain change (time/depth)."""
        self.domain = self.domain_var.get()
        self._display_section()

    def _display_section(self):
        """Display the current section based on view mode."""
        view = self.view_var.get()

        if view == 'inline':
            self._display_inline()
        elif view == 'xline':
            self._display_xline()
        elif view == 'timeslice':
            self._display_timeslice()
        elif view == '2d':
            if self.loader.segy_2d_files:
                last_file = list(self.loader.segy_2d_files.keys())[-1]
                data = self.loader.segy_2d_files[last_file]['data']
                info = self.loader.segy_2d_files[last_file]['info']
                self._display_2d_line(data, info)

    def _clear_figure(self):
        """Clear figure and remove colorbar properly."""
        # Remove existing colorbar if present
        if self.colorbar is not None:
            try:
                self.colorbar.remove()
            except:
                pass
            self.colorbar = None

        # Clear the entire figure and recreate axes
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor(self.colors['surface'])

    def _overlay_faults(self, view_type='inline'):
        """Overlay fault lines on current display."""
        if not self.show_faults.get() or not self.faults:
            return

        for fault in self.faults[:20]:  # Limit to top 20 faults for clarity
            try:
                if view_type == 'inline':
                    # For inline view, show faults as vertical lines at crossline positions
                    # Faults have 'location' with inline/crossline info
                    if 'representative_xl' in fault:
                        xl = fault['representative_xl']
                        self.ax.axvline(x=xl, color='yellow', linewidth=1.5, alpha=0.7, linestyle='--')
                    elif 'location' in fault:
                        xl = fault['location'].get('crossline', fault['location'].get('xl'))
                        if xl:
                            self.ax.axvline(x=xl, color='yellow', linewidth=1.5, alpha=0.7, linestyle='--')

                elif view_type == 'xline':
                    # For crossline view, show faults as vertical lines at inline positions
                    if 'representative_il' in fault:
                        il = fault['representative_il']
                        self.ax.axvline(x=il, color='yellow', linewidth=1.5, alpha=0.7, linestyle='--')
                    elif 'location' in fault:
                        il = fault['location'].get('inline', fault['location'].get('il'))
                        if il:
                            self.ax.axvline(x=il, color='yellow', linewidth=1.5, alpha=0.7, linestyle='--')

                elif view_type == 'timeslice':
                    # For time slice, draw fault traces if we have coordinates
                    if 'trace_points' in fault:
                        points = fault['trace_points']
                        if len(points) > 1:
                            ils = [p['il'] for p in points]
                            xls = [p['xl'] for p in points]
                            self.ax.plot(xls, ils, color='yellow', linewidth=2, alpha=0.8)
            except Exception as e:
                pass  # Skip faults that don't have proper coordinates

        # Add legend if faults shown
        if self.show_faults.get() and self.faults:
            self.ax.plot([], [], color='yellow', linewidth=2, linestyle='--', label=f'Faults ({min(len(self.faults), 20)} shown)')
            self.ax.legend(loc='upper right', facecolor=self.colors['overlay'], edgecolor=self.colors['fg'], labelcolor=self.colors['fg'])

    def _display_inline(self):
        """Display inline section."""
        if self.loader.segy_3d is None:
            self._show_placeholder()
            return

        il = self.current_il
        data = self.loader.get_inline(il)

        if data is None:
            self.status_label.config(text=f"Could not load inline {il}", fg=self.colors['accent'])
            return

        # Clear figure properly (removes old colorbar)
        self._clear_figure()

        # Create axes
        xlines = np.arange(self.loader.xl_min, self.loader.xl_max + 1)
        if self.domain == 'time':
            y_axis = np.arange(0, self.loader.record_length, self.loader.sample_rate)
            y_label = 'TWT (ms)'
        else:
            # Convert to depth
            time_axis = np.arange(0, self.loader.record_length, self.loader.sample_rate)
            y_axis = np.array([self.loader.time_to_depth(t) for t in time_axis])
            y_label = 'Depth (m)'

        # Clip data for display
        clip = np.percentile(np.abs(data), 99)

        # Display
        extent = [xlines[0], xlines[-1], y_axis[-1], y_axis[0]]
        im = self.ax.imshow(
            data.T,
            aspect='auto',
            extent=extent,
            cmap=SEISMIC_CMAP,
            vmin=-clip,
            vmax=clip
        )

        self.ax.set_xlabel('Crossline', color=self.colors['fg'])
        self.ax.set_ylabel(y_label, color=self.colors['fg'])
        self.ax.set_title(f'Inline {il}', color=self.colors['fg'], fontsize=12, fontweight='bold')
        self.ax.tick_params(colors=self.colors['fg'])

        # Overlay faults if enabled
        self._overlay_faults('inline')

        # Colorbar - store reference for removal
        self.colorbar = self.fig.colorbar(im, ax=self.ax, shrink=0.8)
        self.colorbar.set_label('Amplitude', color=self.colors['fg'])
        self.colorbar.ax.tick_params(colors=self.colors['fg'])

        self.fig.tight_layout()
        self.canvas.draw()

    def _display_xline(self):
        """Display crossline section."""
        if self.loader.segy_3d is None:
            self._show_placeholder()
            return

        xl = self.current_xl
        data = self.loader.get_xline(xl)

        if data is None:
            self.status_label.config(text=f"Could not load crossline {xl}", fg=self.colors['accent'])
            return

        # Clear figure properly (removes old colorbar)
        self._clear_figure()

        # Create axes
        inlines = np.arange(self.loader.il_min, self.loader.il_max + 1)
        if self.domain == 'time':
            y_axis = np.arange(0, self.loader.record_length, self.loader.sample_rate)
            y_label = 'TWT (ms)'
        else:
            time_axis = np.arange(0, self.loader.record_length, self.loader.sample_rate)
            y_axis = np.array([self.loader.time_to_depth(t) for t in time_axis])
            y_label = 'Depth (m)'

        clip = np.percentile(np.abs(data), 99)

        extent = [inlines[0], inlines[-1], y_axis[-1], y_axis[0]]
        im = self.ax.imshow(
            data.T,
            aspect='auto',
            extent=extent,
            cmap=SEISMIC_CMAP,
            vmin=-clip,
            vmax=clip
        )

        self.ax.set_xlabel('Inline', color=self.colors['fg'])
        self.ax.set_ylabel(y_label, color=self.colors['fg'])
        self.ax.set_title(f'Crossline {xl}', color=self.colors['fg'], fontsize=12, fontweight='bold')
        self.ax.tick_params(colors=self.colors['fg'])

        # Overlay faults if enabled
        self._overlay_faults('xline')

        # Colorbar - store reference for removal
        self.colorbar = self.fig.colorbar(im, ax=self.ax, shrink=0.8)
        self.colorbar.set_label('Amplitude', color=self.colors['fg'])
        self.colorbar.ax.tick_params(colors=self.colors['fg'])

        self.fig.tight_layout()
        self.canvas.draw()

    def _display_timeslice(self):
        """Display time slice."""
        if self.loader.segy_3d is None:
            self._show_placeholder()
            return

        time_ms = self.current_time

        self.status_label.config(text=f"Extracting time slice at {time_ms:.0f}ms...", fg=self.colors['warning'])
        self.parent.update()

        data = self.loader.get_time_slice(time_ms)

        if data is None:
            self.status_label.config(text=f"Could not extract time slice", fg=self.colors['accent'])
            return

        # Clear figure properly (removes old colorbar)
        self._clear_figure()

        clip = np.percentile(np.abs(data), 99)

        extent = [self.loader.xl_min, self.loader.xl_max, self.loader.il_max, self.loader.il_min]
        im = self.ax.imshow(
            data,
            aspect='auto',
            extent=extent,
            cmap=SEISMIC_CMAP,
            vmin=-clip,
            vmax=clip
        )

        if self.domain == 'depth':
            depth = self.loader.time_to_depth(time_ms)
            title = f'Time Slice at {time_ms:.0f}ms ({depth:.0f}m depth)'
        else:
            title = f'Time Slice at {time_ms:.0f}ms'

        self.ax.set_xlabel('Crossline', color=self.colors['fg'])
        self.ax.set_ylabel('Inline', color=self.colors['fg'])
        self.ax.set_title(title, color=self.colors['fg'], fontsize=12, fontweight='bold')
        self.ax.tick_params(colors=self.colors['fg'])

        # Overlay faults if enabled
        self._overlay_faults('timeslice')

        # Colorbar - store reference for removal
        self.colorbar = self.fig.colorbar(im, ax=self.ax, shrink=0.8)
        self.colorbar.set_label('Amplitude', color=self.colors['fg'])
        self.colorbar.ax.tick_params(colors=self.colors['fg'])

        self.fig.tight_layout()
        self.canvas.draw()

        self.status_label.config(text=f"Displayed time slice at {time_ms:.0f}ms", fg=self.colors['success'])

    def _display_2d_line(self, data, info):
        """Display 2D seismic line."""
        # Clear figure properly (removes old colorbar)
        self._clear_figure()

        n_traces = info['n_traces']
        record_length = info['record_length']
        sample_rate = info['sample_rate']

        if self.domain == 'time':
            y_axis = np.arange(0, record_length, sample_rate)
            y_label = 'TWT (ms)'
        else:
            time_axis = np.arange(0, record_length, sample_rate)
            y_axis = np.array([self.loader.time_to_depth(t) for t in time_axis])
            y_label = 'Depth (m)'

        clip = np.percentile(np.abs(data), 99)

        extent = [0, n_traces, y_axis[-1], y_axis[0]]
        im = self.ax.imshow(
            data,
            aspect='auto',
            extent=extent,
            cmap=SEISMIC_CMAP,
            vmin=-clip,
            vmax=clip
        )

        self.ax.set_xlabel('Trace Number', color=self.colors['fg'])
        self.ax.set_ylabel(y_label, color=self.colors['fg'])
        self.ax.set_title(f"2D Line: {info['filename']}", color=self.colors['fg'], fontsize=12, fontweight='bold')
        self.ax.tick_params(colors=self.colors['fg'])

        # Colorbar - store reference for removal
        self.colorbar = self.fig.colorbar(im, ax=self.ax, shrink=0.8)
        self.colorbar.set_label('Amplitude', color=self.colors['fg'])
        self.colorbar.ax.tick_params(colors=self.colors['fg'])

        self.fig.tight_layout()
        self.canvas.draw()

    def _ai_interpret(self):
        """Send current view to AI for interpretation."""
        if self.ollama is None:
            self.ai_response.delete(1.0, tk.END)
            self.ai_response.insert(tk.END, "AI not connected. Start Ollama first.")
            return

        # Save current figure to temp file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = f.name
            self.fig.savefig(temp_path, facecolor=self.colors['bg'], dpi=150)

        # Get view info for context
        view = self.view_var.get()
        if view == 'inline':
            context = f"This is an inline section (IL {self.current_il}) from a 3D seismic survey in the Bornu Chad Basin, Nigeria."
        elif view == 'xline':
            context = f"This is a crossline section (XL {self.current_xl}) from a 3D seismic survey in the Bornu Chad Basin, Nigeria."
        elif view == 'timeslice':
            context = f"This is a time slice at {self.current_time:.0f}ms TWT from a 3D seismic survey in the Bornu Chad Basin, Nigeria."
        else:
            context = "This is a 2D seismic line from the Bornu Chad Basin, Nigeria."

        prompt = f"""You are an expert seismic interpreter analyzing data from the Bornu Chad Basin.

{context}

The basin contains:
- Chad Formation (Quaternary): shallow lacustrine
- Fika Shale (Turonian): source rock and seal
- Gongila Formation (Cenomanian): secondary reservoir
- Bima Formation (Albian): primary reservoir target

Please analyze this seismic section and describe:
1. Key reflectors and their characteristics (continuity, amplitude, polarity)
2. Any structural features (faults, folds, closures)
3. Potential hydrocarbon indicators (bright spots, flat spots, dim spots)
4. Data quality observations
5. Recommended interpretation focus areas

Be specific and use proper seismic terminology."""

        self.ai_response.delete(1.0, tk.END)
        self.ai_response.insert(tk.END, "Analyzing seismic section...\n")
        self.parent.update()

        # Use vision model if available
        def run_interpretation():
            try:
                if hasattr(self.ollama, 'interpret_image'):
                    response = self.ollama.interpret_image(temp_path, prompt)
                else:
                    response = self.ollama.chat(prompt)

                self.ai_response.delete(1.0, tk.END)
                self.ai_response.insert(tk.END, response)
            except Exception as e:
                self.ai_response.delete(1.0, tk.END)
                self.ai_response.insert(tk.END, f"Error: {str(e)}")
            finally:
                # Clean up temp file
                try:
                    Path(temp_path).unlink()
                except:
                    pass

        # Run in thread to avoid blocking
        thread = threading.Thread(target=run_interpretation)
        thread.start()

    def get_current_figure(self) -> Figure:
        """Return the current matplotlib figure."""
        return self.fig

    def close(self):
        """Clean up resources."""
        self.loader.close()


class WellTiePanel:
    """
    Well-to-Seismic Tie Display Panel.

    Shows:
    - Log curves (GR, AI, RC)
    - Synthetic seismogram
    - Extracted seismic trace
    - Correlation metrics
    """

    # Predefined well locations from well_locations.json
    WELL_LOCATIONS = {
        'BULTE-1': {'il': 5110, 'xl': 5300, 'in_3d': True},
        'HERWA-01': {'il': 5930, 'xl': 5320, 'in_3d': True},
        'KASADE-01': {'il': 5200, 'xl': 4940, 'in_3d': True},
        'MASU-1': {'il': None, 'xl': None, 'in_3d': False},
        'NGAMMAEAST-1': {'il': None, 'xl': None, 'in_3d': False},
        'NGORNORTH-1': {'il': None, 'xl': None, 'in_3d': False}
    }

    def __init__(self, parent_frame, well_tie_validator: WellTieValidator, colors=None):
        self.parent = parent_frame
        self.validator = well_tie_validator
        self.colors = colors or {
            'bg': '#1a1a2e',
            'surface': '#16213e',
            'overlay': '#0f3460',
            'fg': '#e6e6e6',
            'accent': '#e94560',
            'success': '#4ecca3'
        }

        self.fig = None
        self.canvas = None
        self.current_well = None

        # Try to find petrophysics data directory
        self.petro_dir = Path(__file__).parent / 'well_outputs' / 'data'

        self._create_gui()

    def _create_gui(self):
        """Create the well tie panel."""
        # Control frame row 1
        control_frame = tk.Frame(self.parent, bg=self.colors['surface'])
        control_frame.pack(fill=tk.X, padx=5, pady=5)

        tk.Label(
            control_frame,
            text="Well-to-Seismic Tie Validation",
            font=('Segoe UI', 12, 'bold'),
            bg=self.colors['surface'],
            fg=self.colors['accent']
        ).pack(side=tk.LEFT, padx=10)

        # Predefined wells dropdown
        tk.Label(control_frame, text="Select Well:", bg=self.colors['surface'], fg=self.colors['fg']).pack(side=tk.LEFT, padx=5)
        self.preset_combo = ttk.Combobox(control_frame, values=list(self.WELL_LOCATIONS.keys()), width=15)
        self.preset_combo.pack(side=tk.LEFT, padx=5)
        self.preset_combo.bind('<<ComboboxSelected>>', self._on_preset_selected)

        tk.Button(
            control_frame,
            text="Auto-Load",
            command=self._auto_load_well,
            bg=self.colors['accent'],
            fg='white',
            font=('Segoe UI', 9, 'bold')
        ).pack(side=tk.LEFT, padx=5)

        # Manual load
        tk.Button(
            control_frame,
            text="Load CSV...",
            command=self._load_well,
            bg=self.colors['overlay'],
            fg='white'
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            control_frame,
            text="Calculate Tie",
            command=self._calculate_tie,
            bg=self.colors['success'],
            fg='white',
            font=('Segoe UI', 9, 'bold')
        ).pack(side=tk.LEFT, padx=5)

        # Control frame row 2 - Well locations
        control_frame2 = tk.Frame(self.parent, bg=self.colors['surface'])
        control_frame2.pack(fill=tk.X, padx=5, pady=2)

        # Well selection (loaded wells)
        tk.Label(control_frame2, text="Loaded Wells:", bg=self.colors['surface'], fg=self.colors['fg']).pack(side=tk.LEFT, padx=5)
        self.well_combo = ttk.Combobox(control_frame2, values=[], width=15)
        self.well_combo.pack(side=tk.LEFT, padx=5)
        self.well_combo.bind('<<ComboboxSelected>>', self._on_well_selected)

        # IL/XL entry for well location
        tk.Label(control_frame2, text="IL:", bg=self.colors['surface'], fg=self.colors['fg']).pack(side=tk.LEFT, padx=5)
        self.il_entry = tk.Entry(control_frame2, width=8)
        self.il_entry.insert(0, "5110")
        self.il_entry.pack(side=tk.LEFT)

        tk.Label(control_frame2, text="XL:", bg=self.colors['surface'], fg=self.colors['fg']).pack(side=tk.LEFT, padx=5)
        self.xl_entry = tk.Entry(control_frame2, width=8)
        self.xl_entry.insert(0, "5300")
        self.xl_entry.pack(side=tk.LEFT)

        # In 3D indicator
        self.in_3d_label = tk.Label(
            control_frame2,
            text="",
            font=('Segoe UI', 9),
            bg=self.colors['surface'],
            fg=self.colors['success']
        )
        self.in_3d_label.pack(side=tk.LEFT, padx=10)

        # Results label
        self.results_label = tk.Label(
            self.parent,
            text="",
            font=('Segoe UI', 10),
            bg=self.colors['bg'],
            fg=self.colors['fg'],
            wraplength=800
        )
        self.results_label.pack(fill=tk.X, padx=10, pady=5)

        # Figure for display
        display_frame = tk.Frame(self.parent, bg=self.colors['bg'])
        display_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.fig = Figure(figsize=(12, 8), facecolor=self.colors['bg'])
        self.canvas = FigureCanvasTkAgg(self.fig, display_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self._show_instructions()

    def _on_preset_selected(self, event=None):
        """Handle preset well selection - update IL/XL fields."""
        well_name = self.preset_combo.get()
        if well_name in self.WELL_LOCATIONS:
            loc = self.WELL_LOCATIONS[well_name]
            if loc['il'] is not None:
                self.il_entry.delete(0, tk.END)
                self.il_entry.insert(0, str(loc['il']))
            if loc['xl'] is not None:
                self.xl_entry.delete(0, tk.END)
                self.xl_entry.insert(0, str(loc['xl']))

            if loc['in_3d']:
                self.in_3d_label.config(text="Within 3D survey", fg=self.colors['success'])
            else:
                self.in_3d_label.config(text="Outside 3D (use 2D)", fg=self.colors['warning'])

    def _on_well_selected(self, event=None):
        """Handle loaded well selection - update IL/XL if known."""
        well_name = self.well_combo.get()
        # Try to extract well name without prefix
        for preset_name in self.WELL_LOCATIONS:
            if preset_name in well_name:
                loc = self.WELL_LOCATIONS[preset_name]
                if loc['il'] is not None:
                    self.il_entry.delete(0, tk.END)
                    self.il_entry.insert(0, str(loc['il']))
                if loc['xl'] is not None:
                    self.xl_entry.delete(0, tk.END)
                    self.xl_entry.insert(0, str(loc['xl']))

                if loc['in_3d']:
                    self.in_3d_label.config(text="Within 3D survey", fg=self.colors['success'])
                else:
                    self.in_3d_label.config(text="Outside 3D (use 2D)", fg=self.colors['warning'])
                break

    def _auto_load_well(self):
        """Auto-load well from petrophysics directory."""
        well_name = self.preset_combo.get()
        if not well_name:
            self.results_label.config(text="Please select a well from the dropdown first", fg=self.colors['accent'])
            return

        # Look for petrophysics CSV
        petro_file = self.petro_dir / f"petrophysics_{well_name}.csv"

        if petro_file.exists():
            # Extract actual well name (remove 'petrophysics_' prefix)
            result = self.validator.load_well_logs(well_name, str(petro_file))

            if result['status'] == 'success':
                # Update combo
                current_wells = list(self.well_combo['values']) if self.well_combo['values'] else []
                if well_name not in current_wells:
                    current_wells.append(well_name)
                self.well_combo['values'] = current_wells
                self.well_combo.set(well_name)
                self.current_well = well_name

                # Update IL/XL
                self._on_preset_selected()

                cols_str = ', '.join(result.get('columns', []))
                self.results_label.config(
                    text=f"Auto-loaded {well_name}: {result['samples']} samples. Columns: {cols_str}",
                    fg=self.colors['success']
                )
            else:
                self.results_label.config(
                    text=f"Error loading {well_name}: {result['message']}",
                    fg=self.colors['accent']
                )
        else:
            self.results_label.config(
                text=f"Petrophysics file not found: {petro_file}",
                fg=self.colors['accent']
            )

    def _show_instructions(self):
        """Show instructions when no well loaded."""
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor(self.colors['surface'])

        # Check which wells are available
        available_wells = []
        for well_name in self.WELL_LOCATIONS:
            petro_file = self.petro_dir / f"petrophysics_{well_name}.csv"
            if petro_file.exists():
                loc = self.WELL_LOCATIONS[well_name]
                status = "3D" if loc['in_3d'] else "2D only"
                available_wells.append(f"  - {well_name} ({status})")

        wells_text = "\n".join(available_wells) if available_wells else "  No pre-processed wells found"

        ax.text(
            0.5, 0.5,
            "Well-to-Seismic Tie Workflow\n\n"
            "Quick Start:\n"
            "1. Select a well from dropdown and click 'Auto-Load'\n"
            "2. Click 'Calculate Tie' to generate synthetic\n\n"
            f"Available pre-processed wells:\n{wells_text}\n\n"
            "Or load your own CSV with 'Load CSV...'\n\n"
            "Display shows:\n"
            "GR log | AI log | RC series | Synthetic | Seismic trace\n"
            "with correlation coefficient and quality assessment",
            ha='center', va='center',
            fontsize=11,
            color=self.colors['fg'],
            transform=ax.transAxes
        )
        ax.set_xticks([])
        ax.set_yticks([])
        self.canvas.draw()

    def _load_well(self):
        """Load well log data from user-selected CSV."""
        # Default to petrophysics directory if it exists
        initial_dir = str(self.petro_dir) if self.petro_dir.exists() else None

        filepath = filedialog.askopenfilename(
            title="Select Well Log CSV",
            initialdir=initial_dir,
            filetypes=[("CSV files", "*.csv"), ("All files", "*")]
        )

        if filepath:
            # Extract well name - handle 'petrophysics_WELLNAME.csv' format
            filename = Path(filepath).stem
            if filename.startswith('petrophysics_'):
                well_name = filename.replace('petrophysics_', '')
            else:
                well_name = filename

            result = self.validator.load_well_logs(well_name, filepath)

            if result['status'] == 'success':
                # Update combo
                current_wells = list(self.well_combo['values']) if self.well_combo['values'] else []
                if well_name not in current_wells:
                    current_wells.append(well_name)
                self.well_combo['values'] = current_wells
                self.well_combo.set(well_name)
                self.current_well = well_name

                # Try to set IL/XL from known locations
                self._on_well_selected()

                cols_str = ', '.join(result.get('columns', []))
                self.results_label.config(
                    text=f"Loaded {well_name}: {result['samples']} samples. Columns: {cols_str}",
                    fg=self.colors['success']
                )
            else:
                self.results_label.config(
                    text=f"Error: {result['message']}",
                    fg=self.colors['accent']
                )

    def _calculate_tie(self):
        """Calculate synthetic and correlation."""
        well_name = self.well_combo.get()
        if not well_name:
            self.results_label.config(text="Please select a well first", fg=self.colors['accent'])
            return

        # Get IL/XL
        try:
            il = int(self.il_entry.get())
            xl = int(self.xl_entry.get())
        except:
            self.results_label.config(text="Invalid IL/XL values", fg=self.colors['accent'])
            return

        self.results_label.config(text="Calculating synthetic seismogram...", fg=self.colors['warning'])
        self.parent.update()

        # Calculate synthetic (use 12 Hz to match seismic)
        syn_result = self.validator.calculate_synthetic(well_name, wavelet_freq=12.0)

        if syn_result['status'] != 'success':
            self.results_label.config(text=f"Error: {syn_result.get('message', 'Unknown')}", fg=self.colors['accent'])
            return

        # Extract seismic trace
        if self.validator.loader.segy_3d is not None:
            trace_result = self.validator.extract_seismic_trace(well_name, il, xl)

            if trace_result['status'] == 'success':
                # Correlate
                corr_result = self.validator.correlate_well_tie(well_name)

                if corr_result['status'] == 'success':
                    self._display_tie(well_name, corr_result)
                    return

        # If no seismic, just show synthetic
        self._display_synthetic_only(well_name, syn_result)

    def _display_tie(self, well_name: str, corr_result: dict):
        """Display complete well tie with correlation."""
        tie = self.validator.ties[well_name]

        self.fig.clear()

        # Create 5-panel display
        gs = self.fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1.5, 1.5], wspace=0.05)

        df = self.validator.wells[well_name]
        depth = tie['depth']
        twt = tie['twt_ms']

        # Panel 1: GR log
        ax1 = self.fig.add_subplot(gs[0])
        ax1.set_facecolor(self.colors['surface'])
        if 'GR' in df.columns:
            gr = df['GR'].values[:len(depth)]
            ax1.plot(gr, twt, 'g-', lw=0.5)
            ax1.fill_betweenx(twt, 0, gr, alpha=0.3, color='green')
        ax1.set_ylabel('TWT (ms)', color=self.colors['fg'])
        ax1.set_xlabel('GR', color=self.colors['fg'])
        ax1.set_title('Gamma Ray', color=self.colors['fg'], fontsize=10)
        ax1.invert_yaxis()
        ax1.tick_params(colors=self.colors['fg'])
        ax1.set_xlim(0, 150)

        # Panel 2: AI
        ax2 = self.fig.add_subplot(gs[1])
        ax2.set_facecolor(self.colors['surface'])
        ax2.plot(tie['ai'], twt, 'b-', lw=0.5)
        ax2.set_xlabel('AI', color=self.colors['fg'])
        ax2.set_title('Acoustic\nImpedance', color=self.colors['fg'], fontsize=10)
        ax2.invert_yaxis()
        ax2.tick_params(colors=self.colors['fg'])
        ax2.set_yticklabels([])

        # Panel 3: RC
        ax3 = self.fig.add_subplot(gs[2])
        ax3.set_facecolor(self.colors['surface'])
        ax3.plot(tie['rc'], twt, 'k-', lw=0.5)
        ax3.axvline(0, color='gray', lw=0.5)
        ax3.set_xlabel('RC', color=self.colors['fg'])
        ax3.set_title('Reflection\nCoef', color=self.colors['fg'], fontsize=10)
        ax3.invert_yaxis()
        ax3.tick_params(colors=self.colors['fg'])
        ax3.set_yticklabels([])
        ax3.set_xlim(-0.2, 0.2)

        # Panel 4: Synthetic
        ax4 = self.fig.add_subplot(gs[3])
        ax4.set_facecolor(self.colors['surface'])
        syn = tie['synthetic']
        twt_reg = tie['twt_regular']
        ax4.plot(syn, twt_reg, 'r-', lw=1, label='Synthetic')
        ax4.fill_betweenx(twt_reg, 0, syn, where=syn > 0, alpha=0.5, color='red')
        ax4.fill_betweenx(twt_reg, 0, syn, where=syn < 0, alpha=0.5, color='blue')
        ax4.axvline(0, color='gray', lw=0.5)
        ax4.set_xlabel('Amplitude', color=self.colors['fg'])
        ax4.set_title('Synthetic', color=self.colors['fg'], fontsize=10)
        ax4.invert_yaxis()
        ax4.tick_params(colors=self.colors['fg'])
        ax4.set_yticklabels([])

        # Panel 5: Seismic trace
        ax5 = self.fig.add_subplot(gs[4])
        ax5.set_facecolor(self.colors['surface'])

        if 'correlation' in tie:
            seis = tie['correlation']['seismic_interp']
            ax5.plot(seis, twt_reg, 'b-', lw=1, label='Seismic')
            ax5.fill_betweenx(twt_reg, 0, seis, where=seis > 0, alpha=0.5, color='red')
            ax5.fill_betweenx(twt_reg, 0, seis, where=seis < 0, alpha=0.5, color='blue')

        ax5.axvline(0, color='gray', lw=0.5)
        ax5.set_xlabel('Amplitude', color=self.colors['fg'])
        ax5.set_title('Seismic', color=self.colors['fg'], fontsize=10)
        ax5.invert_yaxis()
        ax5.tick_params(colors=self.colors['fg'])
        ax5.set_yticklabels([])

        # Main title with correlation
        corr = corr_result['correlation_coefficient']
        shift = corr_result['optimal_shift_ms']
        phase = corr_result['optimal_phase_deg']
        quality = corr_result['quality']

        title_color = self.colors['success'] if quality in ['EXCELLENT', 'GOOD'] else self.colors['warning']

        self.fig.suptitle(
            f"{well_name} Well Tie - Correlation: r={corr:.3f} ({quality}) | Shift: {shift:.1f}ms | Phase: {phase:.0f}°",
            color=title_color,
            fontsize=12,
            fontweight='bold'
        )

        self.fig.tight_layout()
        self.canvas.draw()

        self.results_label.config(
            text=f"Well tie {quality}: r={corr:.3f}, shift={shift:.1f}ms, phase={phase:.0f}°. "
                 f"Depth: {tie['depth'][0]:.0f}-{tie['depth'][-1]:.0f}m, "
                 f"TWT: {tie['twt_ms'][0]:.0f}-{tie['twt_ms'][-1]:.0f}ms",
            fg=title_color
        )

    def _display_synthetic_only(self, well_name: str, syn_result: dict):
        """Display synthetic without seismic (no seismic loaded)."""
        tie = self.validator.ties[well_name]

        self.fig.clear()

        # Create 4-panel display
        gs = self.fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 2], wspace=0.1)

        df = self.validator.wells[well_name]
        depth = tie['depth']
        twt = tie['twt_ms']

        # Panel 1: GR log
        ax1 = self.fig.add_subplot(gs[0])
        ax1.set_facecolor(self.colors['surface'])
        if 'GR' in df.columns:
            gr_mask = ~np.isnan(df['GR'].values[:len(depth)])
            if np.any(gr_mask):
                ax1.plot(df['GR'].values[:len(depth)], twt, 'g-', lw=0.5)
        ax1.set_ylabel('TWT (ms)', color=self.colors['fg'])
        ax1.set_xlabel('GR', color=self.colors['fg'])
        ax1.set_title('Gamma Ray', color=self.colors['fg'], fontsize=10)
        ax1.invert_yaxis()
        ax1.tick_params(colors=self.colors['fg'])

        # Panel 2: AI
        ax2 = self.fig.add_subplot(gs[1])
        ax2.set_facecolor(self.colors['surface'])
        ax2.plot(tie['ai'], twt, 'b-', lw=0.5)
        ax2.set_xlabel('AI', color=self.colors['fg'])
        ax2.set_title('Acoustic Impedance', color=self.colors['fg'], fontsize=10)
        ax2.invert_yaxis()
        ax2.tick_params(colors=self.colors['fg'])
        ax2.set_yticklabels([])

        # Panel 3: RC
        ax3 = self.fig.add_subplot(gs[2])
        ax3.set_facecolor(self.colors['surface'])
        ax3.plot(tie['rc'], twt, 'k-', lw=0.5)
        ax3.axvline(0, color='gray', lw=0.5)
        ax3.set_xlabel('RC', color=self.colors['fg'])
        ax3.set_title('Reflection Coef', color=self.colors['fg'], fontsize=10)
        ax3.invert_yaxis()
        ax3.tick_params(colors=self.colors['fg'])
        ax3.set_yticklabels([])

        # Panel 4: Synthetic
        ax4 = self.fig.add_subplot(gs[3])
        ax4.set_facecolor(self.colors['surface'])
        syn = tie['synthetic']
        twt_reg = tie['twt_regular']
        ax4.plot(syn, twt_reg, 'r-', lw=1)
        ax4.fill_betweenx(twt_reg, 0, syn, where=syn > 0, alpha=0.5, color='red')
        ax4.fill_betweenx(twt_reg, 0, syn, where=syn < 0, alpha=0.5, color='blue')
        ax4.axvline(0, color='gray', lw=0.5)
        ax4.set_xlabel('Amplitude', color=self.colors['fg'])
        ax4.set_title('Synthetic Seismogram', color=self.colors['fg'], fontsize=10)
        ax4.invert_yaxis()
        ax4.tick_params(colors=self.colors['fg'])
        ax4.set_yticklabels([])

        self.fig.suptitle(
            f"{well_name} Synthetic Seismogram (Load 3D seismic to correlate)",
            color=self.colors['warning'],
            fontsize=12,
            fontweight='bold'
        )

        self.fig.tight_layout()
        self.canvas.draw()

        self.results_label.config(
            text=f"Synthetic generated. Depth: {syn_result['depth_range'][0]:.0f}-{syn_result['depth_range'][1]:.0f}m, "
                 f"TWT: {syn_result['twt_range'][0]:.0f}-{syn_result['twt_range'][1]:.0f}ms. "
                 f"Load 3D seismic to calculate correlation.",
            fg=self.colors['warning']
        )


# Main entry point for standalone testing
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Seismic Viewer - PhD Research")
    root.geometry("1400x900")

    colors = {
        'bg': '#1a1a2e',
        'surface': '#16213e',
        'overlay': '#0f3460',
        'fg': '#e6e6e6',
        'accent': '#e94560',
        'success': '#4ecca3',
        'warning': '#ffc107'
    }
    root.configure(bg=colors['bg'])

    # Create notebook for tabs
    notebook = ttk.Notebook(root)
    notebook.pack(fill=tk.BOTH, expand=True)

    # Seismic viewer tab
    viewer_frame = tk.Frame(notebook, bg=colors['bg'])
    notebook.add(viewer_frame, text="Seismic Viewer")

    viewer = SeismicViewerGUI(viewer_frame, colors=colors)

    # Well tie tab
    tie_frame = tk.Frame(notebook, bg=colors['bg'])
    notebook.add(tie_frame, text="Well Tie Validation")

    tie_panel = WellTiePanel(tie_frame, viewer.well_tie, colors=colors)

    root.mainloop()
