"""
Synthetic Seismogram Generation
================================
Creates well-to-seismic ties - THE FOUNDATION of seismic interpretation.

Without this, your horizons are meaningless.

Author: Moses Ekene Obasi
PhD Research - University of Calabar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.interpolate import interp1d
from pathlib import Path
from typing import Dict, Tuple, Optional
import json

# Try GPU acceleration
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

BASE_DIR = Path(__file__).parent.parent.resolve()

# Import centralized configuration
try:
    import sys
    sys.path.insert(0, str(BASE_DIR))
    from project_config import get_config, ProjectConfig
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False


class SyntheticSeismogram:
    """
    Generate synthetic seismograms from well logs.

    Workflow:
    1. Load DT (sonic) and RHOB (density) logs
    2. Calculate Acoustic Impedance (AI = Velocity * Density)
    3. Calculate Reflection Coefficients (RC)
    4. Extract wavelet from seismic
    5. Convolve RC with wavelet = Synthetic
    6. Correlate synthetic with seismic trace at well
    7. Adjust time-depth relationship
    """

    def __init__(self):
        self.wells = {}
        self.synthetics = {}
        self.wavelets = {}
        self.ties = {}

    def load_well_logs(self, well_name: str, las_file: str = None, csv_file: str = None) -> pd.DataFrame:
        """Load well logs from LAS or CSV file"""
        if csv_file and Path(csv_file).exists():
            df = pd.read_csv(csv_file)
            # Standardize column names
            col_map = {
                'DEPTH': 'Depth_m', 'DEPT': 'Depth_m', 'Depth_m': 'Depth_m',
                'DT': 'DT', 'DT8': 'DT', 'DTCO': 'DT',  # Sonic in us/ft
                'RHOB': 'RHOB', 'DEN': 'RHOB',  # Density in g/cc
                'GR': 'GR', 'Vsh': 'Vsh'
            }
            df = df.rename(columns={c: col_map.get(c, c) for c in df.columns})
            self.wells[well_name] = df
            return df

        elif las_file and Path(las_file).exists():
            import lasio
            las = lasio.read(las_file)
            df = las.df().reset_index()
            df = df.rename(columns={'DEPT': 'Depth_m', 'DEPTH': 'Depth_m'})
            self.wells[well_name] = df
            return df
        else:
            raise FileNotFoundError(f"No valid file found for {well_name}")

    def calculate_acoustic_impedance(self, well_name: str) -> np.ndarray:
        """
        Calculate Acoustic Impedance from sonic and density logs.

        AI = Velocity (m/s) * Density (kg/m³)

        Sonic log is in us/ft, need to convert:
        Velocity (m/s) = 304800 / DT (us/ft)
        """
        df = self.wells[well_name]

        # Map common column name variations
        dt_cols = ['DT', 'DT_usm', 'DT8', 'DTCO', 'Sonic']
        rhob_cols = ['RHOB', 'RHOB_gcc', 'DEN', 'Density']
        vel_cols = ['Velocity_ms', 'VELOCITY', 'Vp']

        dt_col = None
        for col in dt_cols:
            if col in df.columns:
                dt_col = col
                break

        rhob_col = None
        for col in rhob_cols:
            if col in df.columns:
                rhob_col = col
                break

        vel_col = None
        for col in vel_cols:
            if col in df.columns:
                vel_col = col
                break

        # Get sonic and density
        if dt_col is None and vel_col is None:
            raise ValueError(f"Sonic log (DT) not found in {well_name}")
        if rhob_col is None:
            raise ValueError(f"Density log (RHOB) not found in {well_name}")

        rhob = df[rhob_col].values

        # Get or calculate velocity
        if vel_col and vel_col in df.columns:
            velocity = df[vel_col].values
        else:
            dt = df[dt_col].values
            # Handle null values
            dt = np.where((dt <= 0) | (dt > 300) | np.isnan(dt), np.nan, dt)
            # Convert DT to velocity (m/s)
            # DT is in us/ft, so: V = 1e6 / (DT * 3.28084) = 304800 / DT
            velocity = 304800.0 / dt

        # Handle null values for density
        rhob = np.where((rhob <= 0) | (rhob > 3.5) | np.isnan(rhob), np.nan, rhob)  # m/s

        # Convert density to kg/m³
        density = rhob * 1000  # g/cc to kg/m³

        # Calculate AI
        ai = velocity * density

        # Store in dataframe
        df['Velocity_ms'] = velocity
        df['Density_kgm3'] = density
        df['AI'] = ai

        print(f"{well_name}: AI range = {np.nanmin(ai):.0f} - {np.nanmax(ai):.0f}")
        return ai

    def calculate_reflection_coefficients(self, well_name: str) -> np.ndarray:
        """
        Calculate Reflection Coefficients from Acoustic Impedance.

        RC = (AI2 - AI1) / (AI2 + AI1)

        This is the "reflectivity series" that creates seismic reflections.
        """
        df = self.wells[well_name]

        if 'AI' not in df.columns:
            self.calculate_acoustic_impedance(well_name)

        ai = df['AI'].values

        # Calculate RC at each interface
        rc = np.zeros_like(ai)
        for i in range(1, len(ai)):
            if np.isnan(ai[i]) or np.isnan(ai[i-1]):
                rc[i] = 0
            else:
                rc[i] = (ai[i] - ai[i-1]) / (ai[i] + ai[i-1])

        # Clip extreme values
        rc = np.clip(rc, -0.5, 0.5)

        df['RC'] = rc

        print(f"{well_name}: RC range = {np.nanmin(rc):.4f} - {np.nanmax(rc):.4f}")
        return rc

    def create_ricker_wavelet(self, frequency: float = 25.0,
                               sample_rate: float = 0.004,
                               duration: float = 0.128) -> np.ndarray:
        """
        Create a Ricker wavelet (Mexican hat wavelet).

        This is the most common wavelet for synthetic seismograms.

        Parameters:
        - frequency: Dominant frequency in Hz (use your seismic's frequency!)
        - sample_rate: Sampling interval in seconds (typically 0.004 = 4ms)
        - duration: Wavelet length in seconds
        """
        t = np.arange(-duration/2, duration/2, sample_rate)

        # Ricker wavelet formula
        pi2 = (np.pi * frequency * t) ** 2
        wavelet = (1 - 2 * pi2) * np.exp(-pi2)

        # Normalize
        wavelet = wavelet / np.max(np.abs(wavelet))

        self.wavelets['ricker'] = {
            'wavelet': wavelet,
            'frequency': frequency,
            'sample_rate': sample_rate,
            'time': t
        }

        return wavelet

    def extract_wavelet_from_seismic(self, seismic_trace: np.ndarray,
                                      sample_rate: float = 0.004,
                                      window_length: int = 64) -> np.ndarray:
        """
        Extract statistical wavelet from seismic data.

        Uses autocorrelation method - simple but effective for PhD.
        """
        # Autocorrelation
        autocorr = np.correlate(seismic_trace, seismic_trace, mode='full')
        center = len(autocorr) // 2

        # Extract wavelet from center
        half_len = window_length // 2
        wavelet = autocorr[center - half_len:center + half_len]

        # Normalize
        wavelet = wavelet / np.max(np.abs(wavelet))

        self.wavelets['extracted'] = {
            'wavelet': wavelet,
            'sample_rate': sample_rate,
            'method': 'autocorrelation'
        }

        return wavelet

    def depth_to_time(self, well_name: str,
                      velocity_model: str = 'checkshot') -> np.ndarray:
        """
        Convert depth to two-way time (TWT).

        TWT = 2 * integral(dz / V(z))

        For simple case, use average velocity from sonic log.
        """
        df = self.wells[well_name]
        depth = df['Depth_m'].values

        if 'Velocity_ms' not in df.columns:
            self.calculate_acoustic_impedance(well_name)

        velocity = df['Velocity_ms'].values

        # Handle NaN velocities - use interpolation
        valid = ~np.isnan(velocity)
        if valid.sum() < 10:
            # Use constant velocity if not enough data
            velocity = np.full_like(depth, 2500.0)
        else:
            # Interpolate NaN values
            interp_func = interp1d(depth[valid], velocity[valid],
                                   bounds_error=False, fill_value='extrapolate')
            velocity = interp_func(depth)

        # Calculate TWT using cumulative integration
        # TWT = 2 * sum(dz / V)
        dz = np.diff(depth, prepend=depth[0])
        dt = dz / velocity  # One-way time
        twt = 2 * np.cumsum(dt)  # Two-way time in seconds

        df['TWT_s'] = twt
        df['TWT_ms'] = twt * 1000

        print(f"{well_name}: TWT range = {twt.min()*1000:.0f} - {twt.max()*1000:.0f} ms")
        return twt

    def resample_to_seismic(self, well_name: str,
                            sample_rate: float = 0.004,
                            max_time: float = 4.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resample RC series from depth to regular time sampling.

        This is critical - seismic is regularly sampled in time, logs are in depth.
        """
        df = self.wells[well_name]

        if 'TWT_s' not in df.columns:
            self.depth_to_time(well_name)
        if 'RC' not in df.columns:
            self.calculate_reflection_coefficients(well_name)

        twt = df['TWT_s'].values
        rc = df['RC'].values

        # Create regular time sampling
        time_regular = np.arange(0, max_time, sample_rate)

        # Interpolate RC to regular time
        valid = ~(np.isnan(twt) | np.isnan(rc))
        if valid.sum() < 10:
            raise ValueError(f"Not enough valid data in {well_name}")

        interp_func = interp1d(twt[valid], rc[valid],
                               bounds_error=False, fill_value=0)
        rc_resampled = interp_func(time_regular)

        return time_regular, rc_resampled

    def generate_synthetic(self, well_name: str,
                          wavelet_freq: float = 12.0,  # Use YOUR seismic frequency!
                          sample_rate: float = 0.004) -> Dict:
        """
        Generate synthetic seismogram.

        Synthetic = Wavelet * Reflectivity (convolution)
        """
        print(f"\nGenerating synthetic for {well_name}...")

        # Get reflectivity in time
        time, rc = self.resample_to_seismic(well_name, sample_rate)

        # Create wavelet
        wavelet = self.create_ricker_wavelet(frequency=wavelet_freq,
                                             sample_rate=sample_rate)

        # Convolve RC with wavelet
        synthetic = np.convolve(rc, wavelet, mode='same')

        # Normalize
        if np.max(np.abs(synthetic)) > 0:
            synthetic = synthetic / np.max(np.abs(synthetic))

        self.synthetics[well_name] = {
            'time': time,
            'rc': rc,
            'synthetic': synthetic,
            'wavelet_freq': wavelet_freq,
            'sample_rate': sample_rate
        }

        print(f"  Synthetic generated: {len(synthetic)} samples, "
              f"{time[-1]*1000:.0f} ms total")

        return self.synthetics[well_name]

    def correlate_with_seismic(self, well_name: str,
                               seismic_trace: np.ndarray,
                               seismic_time: np.ndarray = None,
                               max_shift_ms: float = 100) -> Dict:
        """
        Cross-correlate synthetic with seismic trace to find best tie.

        Returns:
        - correlation coefficient
        - optimal time shift
        - quality assessment
        """
        if well_name not in self.synthetics:
            raise ValueError(f"Generate synthetic for {well_name} first")

        synthetic = self.synthetics[well_name]['synthetic']
        synth_time = self.synthetics[well_name]['time']
        sample_rate = self.synthetics[well_name]['sample_rate']

        # Ensure same length
        min_len = min(len(synthetic), len(seismic_trace))
        synthetic = synthetic[:min_len]
        seismic = seismic_trace[:min_len]

        # Cross-correlation
        max_lag = int(max_shift_ms / 1000 / sample_rate)
        correlation = signal.correlate(seismic, synthetic, mode='full')
        lags = signal.correlation_lags(len(seismic), len(synthetic), mode='full')

        # Find best correlation within allowed shift
        valid = np.abs(lags) <= max_lag
        best_idx = np.argmax(np.abs(correlation[valid]))
        best_lag = lags[valid][best_idx]
        best_corr = correlation[valid][best_idx]

        # Normalize correlation
        norm_corr = best_corr / (np.std(synthetic) * np.std(seismic) * min_len)

        # Quality assessment
        if abs(norm_corr) > 0.7:
            quality = 'EXCELLENT'
        elif abs(norm_corr) > 0.5:
            quality = 'GOOD'
        elif abs(norm_corr) > 0.3:
            quality = 'FAIR'
        else:
            quality = 'POOR'

        shift_ms = best_lag * sample_rate * 1000

        self.ties[well_name] = {
            'correlation': round(float(norm_corr), 3),
            'shift_ms': round(float(shift_ms), 1),
            'quality': quality,
            'shift_samples': int(best_lag)
        }

        print(f"  Tie quality: {quality} (r={norm_corr:.3f}, shift={shift_ms:.1f} ms)")

        return self.ties[well_name]

    def plot_synthetic_tie(self, well_name: str,
                           seismic_trace: np.ndarray = None,
                           save_path: str = None):
        """
        Plot synthetic seismogram with well logs and seismic trace.

        This is THE KEY FIGURE for your thesis Chapter 4!
        """
        if well_name not in self.synthetics:
            raise ValueError(f"Generate synthetic for {well_name} first")

        df = self.wells[well_name]
        synth_data = self.synthetics[well_name]

        fig, axes = plt.subplots(1, 5, figsize=(15, 10), sharey=True)
        fig.suptitle(f'Well-to-Seismic Tie: {well_name}', fontsize=14, fontweight='bold')

        # Use TWT for y-axis
        if 'TWT_ms' in df.columns:
            depth_col = 'TWT_ms'
            ylabel = 'TWT (ms)'
        else:
            depth_col = 'Depth_m'
            ylabel = 'Depth (m)'

        # 1. GR log
        ax = axes[0]
        if 'GR' in df.columns:
            ax.plot(df['GR'], df[depth_col], 'g-', linewidth=0.5)
            ax.fill_betweenx(df[depth_col], 0, df['GR'],
                            where=df['GR'] > 75, color='brown', alpha=0.5)
            ax.fill_betweenx(df[depth_col], 0, df['GR'],
                            where=df['GR'] <= 75, color='yellow', alpha=0.5)
        ax.set_xlabel('GR (API)')
        ax.set_xlim(0, 150)
        ax.set_title('Gamma Ray')

        # 2. Sonic log
        ax = axes[1]
        if 'DT' in df.columns:
            ax.plot(df['DT'], df[depth_col], 'b-', linewidth=0.5)
        ax.set_xlabel('DT (us/ft)')
        ax.set_xlim(40, 140)
        ax.invert_xaxis()
        ax.set_title('Sonic')

        # 3. Density log
        ax = axes[2]
        if 'RHOB' in df.columns:
            ax.plot(df['RHOB'], df[depth_col], 'r-', linewidth=0.5)
        ax.set_xlabel('RHOB (g/cc)')
        ax.set_xlim(1.8, 2.8)
        ax.set_title('Density')

        # 4. Acoustic Impedance
        ax = axes[3]
        if 'AI' in df.columns:
            ax.plot(df['AI']/1e6, df[depth_col], 'purple', linewidth=0.5)
        ax.set_xlabel('AI (x10^6)')
        ax.set_title('Impedance')

        # 5. Synthetic vs Seismic
        ax = axes[4]
        time_ms = synth_data['time'] * 1000

        # Plot synthetic
        ax.plot(synth_data['synthetic'], time_ms, 'b-', linewidth=1, label='Synthetic')
        ax.fill_betweenx(time_ms, 0, synth_data['synthetic'],
                        where=synth_data['synthetic'] > 0, color='blue', alpha=0.3)
        ax.fill_betweenx(time_ms, 0, synth_data['synthetic'],
                        where=synth_data['synthetic'] < 0, color='red', alpha=0.3)

        # Plot seismic if provided
        if seismic_trace is not None:
            seismic_time = np.arange(len(seismic_trace)) * synth_data['sample_rate'] * 1000
            ax.plot(seismic_trace[:len(time_ms)] * 0.8 + 1.5, time_ms, 'k-',
                   linewidth=0.8, label='Seismic')

        ax.set_xlabel('Amplitude')
        ax.set_title('Synthetic | Seismic')
        ax.legend(loc='upper right', fontsize=8)
        ax.axvline(x=0, color='gray', linewidth=0.5)

        # Set y-axis
        axes[0].set_ylabel(ylabel)
        axes[0].invert_yaxis()

        # Add tie quality if available
        if well_name in self.ties:
            tie = self.ties[well_name]
            fig.text(0.5, 0.02,
                    f"Tie Quality: {tie['quality']} | Correlation: {tie['correlation']:.3f} | "
                    f"Shift: {tie['shift_ms']:.1f} ms",
                    ha='center', fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='wheat'))

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.08)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")

        plt.close()
        return fig

    def identify_formation_tops(self, well_name: str,
                                formations: Dict[str, float] = None) -> Dict:
        """
        Identify formation tops in TWT from known depth picks.

        This ties your geological formations to seismic time.
        Now loads REAL picks from formation_tops.json instead of placeholders!
        """
        df = self.wells[well_name]

        if 'TWT_ms' not in df.columns:
            self.depth_to_time(well_name)
            df = self.wells[well_name]  # Refresh after depth-to-time conversion

        # Load REAL formation tops from file (not placeholders!)
        if formations is None:
            formations = self._load_formation_tops_for_well(well_name)

        tops_in_time = {}

        for fm_name, fm_depth in formations.items():
            # Find closest depth in well
            if fm_depth is None:
                continue  # Skip formations not present in this well

            depth_col = 'Depth_m' if 'Depth_m' in df.columns else 'DEPT'
            idx = np.abs(df[depth_col] - fm_depth).argmin()
            twt_ms = df.loc[idx, 'TWT_ms'] if 'TWT_ms' in df.columns else 0.0

            if not np.isnan(twt_ms):
                tops_in_time[fm_name] = {
                    'depth_m': fm_depth,
                    'twt_ms': round(float(twt_ms), 1),
                    'well': well_name
                }

        return tops_in_time

    def _load_formation_tops_for_well(self, well_name: str) -> Dict[str, float]:
        """Load formation tops from formation_tops.json"""
        tops_file = BASE_DIR.parent / "formation_tops.json"

        formations = {}

        if tops_file.exists():
            try:
                with open(tops_file, 'r') as f:
                    data = json.load(f)

                wells_data = data.get('wells', {})

                # Try exact match first
                well_data = wells_data.get(well_name, {})

                # Try without hyphen if not found
                if not well_data:
                    for wn, wd in wells_data.items():
                        if wn.replace("-", "").upper() == well_name.replace("-", "").upper():
                            well_data = wd
                            break

                if well_data:
                    for fm in ['Chad_Fm', 'Fika_Shale', 'Gongila_Fm', 'Bima_Sst']:
                        value = well_data.get(fm)
                        if value is not None:
                            formations[fm] = value
                    print(f"  Loaded REAL formation tops for {well_name} from formation_tops.json")
                else:
                    print(f"  WARNING: No formation tops found for {well_name} in formation_tops.json")

            except Exception as e:
                print(f"  Error loading formation_tops.json: {e}")

        if not formations:
            print(f"  FALLBACK: Using literature-based defaults for {well_name}")
            # Fallback to literature-based estimates (better than arbitrary placeholders)
            literature_tops = {
                'BULTE-1': {'Chad_Fm': 380, 'Fika_Shale': 720, 'Gongila_Fm': 1050, 'Bima_Sst': 1280},
                'HERWA-01': {'Chad_Fm': 450, 'Fika_Shale': 890, 'Gongila_Fm': 1350, 'Bima_Sst': 1680},
                'MASU-1': {'Fika_Shale': 2025, 'Gongila_Fm': 2450, 'Bima_Sst': 2850},
                'KASADE-01': {'Chad_Fm': 350, 'Fika_Shale': 680, 'Gongila_Fm': 980, 'Bima_Sst': 1250},
                'NGAMMAEAST-1': {'Chad_Fm': 320, 'Fika_Shale': 850, 'Gongila_Fm': 1600, 'Bima_Sst': 2100},
                'NGORNORTH-1': {'Chad_Fm': 280, 'Fika_Shale': 520, 'Gongila_Fm': 850, 'Bima_Sst': 1100}
            }
            # Find matching well
            for wn, tops in literature_tops.items():
                if wn.replace("-", "").upper() == well_name.replace("-", "").upper():
                    formations = tops
                    break

        return formations

    def generate_report(self) -> str:
        """Generate synthetic seismogram report"""
        lines = []
        lines.append("=" * 70)
        lines.append("SYNTHETIC SEISMOGRAM REPORT")
        lines.append("Well-to-Seismic Tie Analysis")
        lines.append("=" * 70)

        for well_name in self.synthetics.keys():
            lines.append(f"\nWELL: {well_name}")
            lines.append("-" * 50)

            df = self.wells[well_name]
            synth = self.synthetics[well_name]

            lines.append(f"  Depth range: {df['Depth_m'].min():.0f} - {df['Depth_m'].max():.0f} m")
            if 'TWT_ms' in df.columns:
                lines.append(f"  TWT range: {df['TWT_ms'].min():.0f} - {df['TWT_ms'].max():.0f} ms")
            lines.append(f"  Wavelet frequency: {synth['wavelet_freq']} Hz")

            if well_name in self.ties:
                tie = self.ties[well_name]
                lines.append(f"  Tie correlation: {tie['correlation']:.3f}")
                lines.append(f"  Time shift: {tie['shift_ms']:.1f} ms")
                lines.append(f"  Quality: {tie['quality']}")

        return "\n".join(lines)

    def save_results(self, output_dir: str):
        """Save all results to JSON"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results = {
            'synthetics': {},
            'ties': self.ties,
            'formation_tops': {}
        }

        for well_name in self.synthetics.keys():
            synth = self.synthetics[well_name]
            results['synthetics'][well_name] = {
                'wavelet_freq': synth['wavelet_freq'],
                'sample_rate': synth['sample_rate'],
                'n_samples': len(synth['synthetic'])
            }

            # Get formation tops
            tops = self.identify_formation_tops(well_name)
            results['formation_tops'][well_name] = tops

        with open(output_path / 'synthetic_results.json', 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to {output_path / 'synthetic_results.json'}")


def run_synthetic_workflow():
    """Run the complete synthetic seismogram workflow"""
    print("=" * 70)
    print("SYNTHETIC SEISMOGRAM WORKFLOW")
    print("=" * 70)

    synth = SyntheticSeismogram()

    # Get paths from config or use defaults
    if CONFIG_AVAILABLE:
        config = get_config()
        output_base = Path(config.output_directory) if config.output_directory else BASE_DIR
        well_data_dir = output_base / "well_outputs" / "data"
        las_dir = Path(config.well_logs_directory) if config.well_logs_directory else Path()
        output_dir = output_base / "interpretation" / "synthetic_outputs"
    else:
        well_data_dir = BASE_DIR / "well_outputs" / "data"
        las_dir = Path()  # Will need to be configured
        output_dir = BASE_DIR / "interpretation" / "synthetic_outputs"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Process wells with both DT and RHOB
    wells_to_process = ['BULTE-1', 'HERWA-01', 'MASU-1', 'KASADE-01']

    for well_name in wells_to_process:
        print(f"\n{'='*50}")
        print(f"Processing {well_name}")
        print('='*50)

        try:
            # Try loading from CSV first (has more processed data)
            csv_file = well_data_dir / f"petrophysics_{well_name}.csv"
            if csv_file.exists():
                df = pd.read_csv(csv_file)
                # Check if we have DT
                dt_cols = [c for c in df.columns if 'DT' in c.upper() or 'SONIC' in c.upper()]
                if not dt_cols:
                    # Load from LAS to get DT
                    las_file = las_dir / f"{well_name}.las"
                    if las_file.exists():
                        import lasio
                        las = lasio.read(str(las_file))
                        las_df = las.df().reset_index()
                        # Merge DT into petrophysics
                        for col in ['DT', 'DT8', 'DTCO']:
                            if col in las_df.columns:
                                # Need to align by depth
                                df['DT'] = np.interp(df['Depth_m'],
                                                     las_df['DEPT'].values if 'DEPT' in las_df.columns else las_df.index,
                                                     las_df[col].values)
                                break

                synth.wells[well_name] = df
            else:
                # Load from LAS
                las_file = las_dir / f"{well_name}.las"
                synth.load_well_logs(well_name, las_file=str(las_file))

            # Generate synthetic (use 12 Hz - your seismic's dominant frequency!)
            synth.generate_synthetic(well_name, wavelet_freq=12.0)

            # Plot and save
            synth.plot_synthetic_tie(
                well_name,
                save_path=str(output_dir / f"synthetic_{well_name}.png")
            )

            # Get formation tops
            tops = synth.identify_formation_tops(well_name)
            print(f"  Formation tops in TWT:")
            for fm, data in tops.items():
                print(f"    {fm}: {data['depth_m']}m = {data['twt_ms']} ms")

        except Exception as e:
            print(f"  Error: {e}")
            continue

    # Generate and save report
    report = synth.generate_report()
    print("\n" + report)

    with open(output_dir / "synthetic_report.txt", 'w') as f:
        f.write(report)

    synth.save_results(str(output_dir))

    print("\n" + "=" * 70)
    print("SYNTHETIC WORKFLOW COMPLETE")
    print("=" * 70)

    return synth


if __name__ == "__main__":
    run_synthetic_workflow()
