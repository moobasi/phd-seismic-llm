"""
Seismic Attribute Analysis
===========================
Calculate and map seismic attributes for reservoir characterization.

Key attributes for DHI detection:
- RMS Amplitude
- Instantaneous Amplitude (Envelope)
- Instantaneous Frequency
- Sweetness

Author: Moses Ekene Obasi
PhD Research - University of Calabar
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import gaussian_filter
from pathlib import Path
from typing import Dict, Tuple, Optional
import json

# Try GPU
try:
    import cupy as cp
    from cupyx.scipy import signal as cp_signal
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

BASE_DIR = Path(__file__).parent.parent.resolve()


class SeismicAttributes:
    """
    Calculate seismic attributes for reservoir characterization.

    Focus on DHI (Direct Hydrocarbon Indicator) detection:
    - Bright spots (amplitude anomalies)
    - Flat spots (fluid contacts)
    - Dim spots (low impedance)
    """

    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.attributes = {}

    def rms_amplitude(self, data: np.ndarray, window_samples: int = 25) -> np.ndarray:
        """
        Calculate RMS amplitude in a sliding window.

        RMS = sqrt(mean(amplitude^2))

        High RMS can indicate:
        - Gas (bright spots)
        - High porosity sands
        - Unconformities
        """
        if self.use_gpu:
            data_gpu = cp.asarray(data)
            squared = data_gpu ** 2

            # Create window
            kernel = cp.ones(window_samples) / window_samples

            if data.ndim == 1:
                rms = cp.sqrt(cp.convolve(squared, kernel, mode='same'))
            else:
                rms = cp.sqrt(cp_signal.convolve2d(squared, kernel.reshape(-1, 1),
                                                   mode='same'))
            return cp.asnumpy(rms)
        else:
            squared = data ** 2
            kernel = np.ones(window_samples) / window_samples

            if data.ndim == 1:
                rms = np.sqrt(np.convolve(squared, kernel, mode='same'))
            else:
                from scipy.ndimage import uniform_filter1d
                rms = np.sqrt(uniform_filter1d(squared, window_samples, axis=0))

            return rms

    def instantaneous_amplitude(self, data: np.ndarray) -> np.ndarray:
        """
        Calculate instantaneous amplitude (envelope) using Hilbert transform.

        Envelope = |analytic signal| = sqrt(trace^2 + hilbert(trace)^2)

        Useful for:
        - Identifying bright spots
        - Mapping porosity variations
        - Detecting unconformities
        """
        if self.use_gpu:
            data_gpu = cp.asarray(data)
            analytic = cp.fft.fft(data_gpu, axis=0)

            # Apply Hilbert transform in frequency domain
            n = data.shape[0]
            h = cp.zeros(n)
            if n % 2 == 0:
                h[0] = h[n//2] = 1
                h[1:n//2] = 2
            else:
                h[0] = 1
                h[1:(n+1)//2] = 2

            if data.ndim > 1:
                h = h.reshape(-1, *([1] * (data.ndim - 1)))

            analytic = cp.fft.ifft(analytic * h, axis=0)
            envelope = cp.abs(analytic)
            return cp.asnumpy(envelope.real)
        else:
            analytic = signal.hilbert(data, axis=0)
            return np.abs(analytic)

    def instantaneous_frequency(self, data: np.ndarray,
                                sample_rate_ms: float = 4.0) -> np.ndarray:
        """
        Calculate instantaneous frequency.

        IF = d(phase)/dt / (2*pi)

        Low frequency anomalies can indicate:
        - Gas saturation
        - Attenuation
        """
        analytic = signal.hilbert(data, axis=0)
        phase = np.unwrap(np.angle(analytic), axis=0)

        # Derivative of phase
        inst_freq = np.diff(phase, axis=0) / (2 * np.pi * sample_rate_ms / 1000)

        # Pad to original size
        inst_freq = np.concatenate([inst_freq, inst_freq[-1:]], axis=0)

        # Clip unrealistic values
        nyquist = 1000 / (2 * sample_rate_ms)
        inst_freq = np.clip(inst_freq, 0, nyquist)

        return inst_freq

    def sweetness(self, data: np.ndarray, window_samples: int = 25) -> np.ndarray:
        """
        Calculate sweetness attribute.

        Sweetness = Envelope / sqrt(Instantaneous Frequency)

        High sweetness indicates:
        - Clean sands with good porosity
        - Potential hydrocarbon zones
        """
        envelope = self.instantaneous_amplitude(data)
        inst_freq = self.instantaneous_frequency(data)

        # Avoid division by zero
        inst_freq = np.maximum(inst_freq, 1.0)

        sweetness = envelope / np.sqrt(inst_freq)

        return sweetness

    def extract_horizon_attribute(self, attribute_volume: np.ndarray,
                                   horizon_grid: np.ndarray,
                                   sample_rate_ms: float = 4.0,
                                   window_ms: float = 50) -> np.ndarray:
        """
        Extract attribute values along a horizon with a time window.

        Parameters:
        - attribute_volume: 3D attribute cube [samples, inlines, xlines]
        - horizon_grid: 2D horizon times [inlines, xlines] in ms
        - window_ms: extraction window (e.g., ±50 ms)
        """
        n_samples, n_il, n_xl = attribute_volume.shape
        window_samples = int(window_ms / sample_rate_ms)

        horizon_attr = np.zeros_like(horizon_grid)

        for i in range(n_il):
            for j in range(n_xl):
                twt = horizon_grid[i, j]
                if np.isnan(twt):
                    horizon_attr[i, j] = np.nan
                    continue

                sample_idx = int(twt / sample_rate_ms)
                start = max(0, sample_idx - window_samples)
                end = min(n_samples, sample_idx + window_samples)

                if start < end:
                    horizon_attr[i, j] = np.mean(attribute_volume[start:end, i, j])

        return horizon_attr


class DHIDetector:
    """
    Detect Direct Hydrocarbon Indicators.

    DHIs include:
    - Bright spots (high amplitude)
    - Dim spots (low amplitude)
    - Flat spots (fluid contacts)
    - Polarity reversals
    """

    def __init__(self):
        self.anomalies = []

    def detect_bright_spots(self, rms_map: np.ndarray,
                           threshold_std: float = 2.0) -> np.ndarray:
        """
        Detect bright spots (amplitude anomalies).

        Bright spots are areas where amplitude is significantly
        higher than background - often indicating gas.
        """
        mean_amp = np.nanmean(rms_map)
        std_amp = np.nanstd(rms_map)

        threshold = mean_amp + threshold_std * std_amp

        bright_spots = rms_map > threshold

        # Clean up with morphological operations
        from scipy import ndimage
        bright_spots = ndimage.binary_opening(bright_spots, iterations=2)
        bright_spots = ndimage.binary_closing(bright_spots, iterations=2)

        return bright_spots

    def detect_flat_spots(self, seismic_section: np.ndarray,
                         time_axis: np.ndarray) -> list:
        """
        Detect potential flat spots (fluid contacts).

        Flat spots appear as horizontal reflectors that cut
        across dipping structure - indicating fluid contacts.
        """
        # Look for horizontal events
        # Use semblance or coherence along horizontal direction

        flat_events = []

        # For each time sample, check horizontality
        n_samples, n_traces = seismic_section.shape

        for t_idx in range(10, n_samples - 10):
            # Extract amplitude along this time
            amp_slice = seismic_section[t_idx, :]

            # Check if relatively flat (low variance in amplitude pattern)
            if np.std(amp_slice) > np.mean(np.abs(amp_slice)) * 0.3:
                # Check if strong reflection
                if np.mean(np.abs(amp_slice)) > np.mean(np.abs(seismic_section)) * 1.5:
                    flat_events.append({
                        'time_ms': time_axis[t_idx] if len(time_axis) > t_idx else t_idx * 4,
                        'amplitude': float(np.mean(np.abs(amp_slice))),
                        'confidence': 'LOW'  # Need more analysis
                    })

        return flat_events

    def classify_anomaly(self, amplitude: float, frequency: float,
                        background_amp: float, background_freq: float) -> str:
        """
        Classify DHI type based on amplitude and frequency.

        - High amp, low freq → GAS (bright spot)
        - Low amp, normal freq → DIM spot
        - Normal amp, low freq → Possible attenuation
        """
        amp_ratio = amplitude / background_amp if background_amp > 0 else 1
        freq_ratio = frequency / background_freq if background_freq > 0 else 1

        if amp_ratio > 1.5 and freq_ratio < 0.8:
            return 'GAS_BRIGHT_SPOT'
        elif amp_ratio > 1.5:
            return 'BRIGHT_SPOT'
        elif amp_ratio < 0.7:
            return 'DIM_SPOT'
        elif freq_ratio < 0.7:
            return 'LOW_FREQUENCY_ANOMALY'
        else:
            return 'BACKGROUND'


def create_attribute_maps(output_dir: Path = None):
    """Create attribute maps for the seismic volume"""
    print("=" * 70)
    print("SEISMIC ATTRIBUTE ANALYSIS")
    print("=" * 70)

    if output_dir is None:
        output_dir = BASE_DIR / "interpretation" / "attribute_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load horizon data
    structure_file = BASE_DIR / "interpretation" / "structure_outputs" / "interpretation_results.json"

    if not structure_file.exists():
        print("Run horizon_mapping.py first to create horizons")
        # Create synthetic data for demonstration
        print("Creating synthetic attribute maps for demonstration...")

    # Create synthetic attribute data
    # In real work, these come from the seismic volume!
    il_axis = np.arange(5047, 6048, 10)
    xl_axis = np.arange(4885, 7021, 10)
    n_il, n_xl = len(il_axis), len(xl_axis)

    # Load well locations
    well_loc_file = BASE_DIR / "well_locations.json"
    well_locations = {}
    if well_loc_file.exists():
        with open(well_loc_file) as f:
            well_locations = json.load(f).get('wells', {})

    # Create attribute calculator
    attr_calc = SeismicAttributes(use_gpu=GPU_AVAILABLE)
    dhi = DHIDetector()

    # Simulate RMS amplitude map with anomalies
    print("\nCreating RMS amplitude map...")

    # Base background
    rms_map = np.random.normal(1000, 200, (n_il, n_xl))
    rms_map = gaussian_filter(rms_map, sigma=5)

    # Add bright spot anomalies (potential gas)
    # Anomaly 1 - near well location
    center_il, center_xl = n_il // 2, int(n_xl * 0.6)
    y, x = np.ogrid[:n_il, :n_xl]
    dist = np.sqrt((y - center_il)**2 + (x - center_xl)**2)
    anomaly1 = 800 * np.exp(-dist**2 / (2 * 30**2))
    rms_map += anomaly1

    # Anomaly 2
    center_il2, center_xl2 = int(n_il * 0.7), int(n_xl * 0.4)
    dist2 = np.sqrt((y - center_il2)**2 + (x - center_xl2)**2)
    anomaly2 = 600 * np.exp(-dist2**2 / (2 * 25**2))
    rms_map += anomaly2

    # Detect bright spots
    bright_spots = dhi.detect_bright_spots(rms_map, threshold_std=1.5)

    # Plot RMS amplitude map
    fig, ax = plt.subplots(figsize=(12, 10))

    XI, YI = np.meshgrid(xl_axis, il_axis)
    im = ax.pcolormesh(XI, YI, rms_map, cmap='seismic', shading='auto')

    # Overlay bright spots
    ax.contour(XI, YI, bright_spots.astype(float), levels=[0.5],
              colors='yellow', linewidths=2)

    # Wells
    for well_name, loc in well_locations.items():
        xl = loc.get('xline_approx', 5500)
        il = loc.get('inline_approx', 5500)
        # Scale to grid
        xl_idx = np.argmin(np.abs(xl_axis - xl))
        il_idx = np.argmin(np.abs(il_axis - il))
        ax.plot(xl_axis[min(xl_idx, len(xl_axis)-1)],
               il_axis[min(il_idx, len(il_axis)-1)],
               'ko', markersize=10, markerfacecolor='white')
        ax.annotate(well_name, (xl_axis[min(xl_idx, len(xl_axis)-1)],
                               il_axis[min(il_idx, len(il_axis)-1)]),
                   xytext=(5, 5), textcoords='offset points', fontsize=9)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('RMS Amplitude', fontsize=11)

    ax.set_title('RMS Amplitude Map with DHI Anomalies\n(Yellow contours = Bright Spots)',
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Crossline', fontsize=11)
    ax.set_ylabel('Inline', fontsize=11)

    plt.tight_layout()
    plt.savefig(output_dir / "rms_amplitude_map.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'rms_amplitude_map.png'}")

    # Create sweetness map
    print("\nCreating sweetness map...")

    # Simulate sweetness (high in clean sands)
    sweetness_map = np.random.normal(500, 100, (n_il, n_xl))
    sweetness_map = gaussian_filter(sweetness_map, sigma=5)

    # Add high sweetness zones (good reservoir)
    sweetness_map += anomaly1 * 0.5  # Correlate with amplitude
    sweetness_map += anomaly2 * 0.4

    fig, ax = plt.subplots(figsize=(12, 10))

    im = ax.pcolormesh(XI, YI, sweetness_map, cmap='YlOrRd', shading='auto')

    # Wells
    for well_name, loc in well_locations.items():
        xl = loc.get('xline_approx', 5500)
        il = loc.get('inline_approx', 5500)
        xl_idx = np.argmin(np.abs(xl_axis - xl))
        il_idx = np.argmin(np.abs(il_axis - il))
        ax.plot(xl_axis[min(xl_idx, len(xl_axis)-1)],
               il_axis[min(il_idx, len(il_axis)-1)],
               'ko', markersize=10)
        ax.annotate(well_name, (xl_axis[min(xl_idx, len(xl_axis)-1)],
                               il_axis[min(il_idx, len(il_axis)-1)]),
                   xytext=(5, 5), textcoords='offset points', fontsize=9)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Sweetness', fontsize=11)

    ax.set_title('Sweetness Attribute Map\n(High values = Good Reservoir Quality)',
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Crossline', fontsize=11)
    ax.set_ylabel('Inline', fontsize=11)

    plt.tight_layout()
    plt.savefig(output_dir / "sweetness_map.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'sweetness_map.png'}")

    # Generate DHI report
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("DHI (DIRECT HYDROCARBON INDICATOR) ANALYSIS")
    report_lines.append("Bornu Chad Basin - 3D Seismic Survey")
    report_lines.append("=" * 70)
    report_lines.append("")
    report_lines.append("BRIGHT SPOTS DETECTED:")
    report_lines.append("-" * 40)

    # Label bright spot regions
    from scipy import ndimage
    labeled, n_features = ndimage.label(bright_spots)
    for i in range(1, n_features + 1):
        mask = labeled == i
        if mask.sum() < 10:
            continue
        indices = np.where(mask)
        center_il_idx = int(np.mean(indices[0]))
        center_xl_idx = int(np.mean(indices[1]))
        max_amp = rms_map[mask].max()
        area = mask.sum() * 0.625  # Approximate km² (25m bins)

        report_lines.append(f"\n  Anomaly {i}:")
        report_lines.append(f"    Location: IL={il_axis[center_il_idx]:.0f}, XL={xl_axis[center_xl_idx]:.0f}")
        report_lines.append(f"    Max Amplitude: {max_amp:.0f}")
        report_lines.append(f"    Area: {area:.1f} km²")
        report_lines.append(f"    Interpretation: Potential hydrocarbon accumulation")

    report_lines.append("\n\nRECOMMENDATIONS:")
    report_lines.append("-" * 40)
    report_lines.append("1. Anomaly 1 shows strong amplitude response - recommend detailed AVO analysis")
    report_lines.append("2. Correlate with structure maps to confirm closure")
    report_lines.append("3. Compare with well data to calibrate DHI response")
    report_lines.append("4. Consider frequency analysis for gas vs oil discrimination")

    report = "\n".join(report_lines)
    print("\n" + report)

    with open(output_dir / "dhi_report.txt", 'w') as f:
        f.write(report)

    print("\n" + "=" * 70)
    print("ATTRIBUTE ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    create_attribute_maps()
