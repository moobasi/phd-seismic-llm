"""
GPU Utilities for Seismic Processing
=====================================

Provides GPU-accelerated operations for seismic data processing.
Falls back to CPU (NumPy) if CuPy is not available.

Author: Moses Ekene Obasi
"""

import numpy as np
from typing import Tuple, Optional, Union
import warnings

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    GPU_AVAILABLE = True

    # Warm up GPU on import
    _warmup = cp.zeros(100)
    del _warmup
    cp.cuda.Stream.null.synchronize()

except ImportError:
    GPU_AVAILABLE = False
    cp = None


def get_array_module(use_gpu: bool = True):
    """Get the appropriate array module (cupy or numpy)"""
    if use_gpu and GPU_AVAILABLE:
        return cp
    return np


def to_gpu(array: np.ndarray) -> Union[np.ndarray, 'cp.ndarray']:
    """Transfer array to GPU if available"""
    if GPU_AVAILABLE:
        return cp.asarray(array)
    return array


def to_cpu(array) -> np.ndarray:
    """Transfer array to CPU"""
    if GPU_AVAILABLE and isinstance(array, cp.ndarray):
        return cp.asnumpy(array)
    return np.asarray(array)


def gpu_info() -> dict:
    """Get GPU information"""
    info = {
        "available": GPU_AVAILABLE,
        "device_name": "N/A",
        "memory_total_gb": 0,
        "memory_free_gb": 0,
        "cupy_version": None
    }

    if GPU_AVAILABLE:
        info["cupy_version"] = cp.__version__
        try:
            device = cp.cuda.Device()
            info["device_name"] = f"GPU {device.id}"
            mem = device.mem_info
            info["memory_total_gb"] = mem[1] / (1024**3)
            info["memory_free_gb"] = mem[0] / (1024**3)
        except:
            pass

    return info


# =============================================================================
# GPU-Accelerated Seismic Operations
# =============================================================================

class GPUSeismicOps:
    """GPU-accelerated seismic processing operations"""

    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.xp = cp if self.use_gpu else np

        if self.use_gpu:
            print(f"GPU acceleration enabled: {gpu_info()['device_name']}")
        else:
            print("Running on CPU")

    def fft(self, data: np.ndarray, axis: int = -1) -> np.ndarray:
        """GPU-accelerated FFT"""
        if self.use_gpu:
            data_gpu = cp.asarray(data)
            result = cp.fft.fft(data_gpu, axis=axis)
            return cp.asnumpy(result)
        return np.fft.fft(data, axis=axis)

    def ifft(self, data: np.ndarray, axis: int = -1) -> np.ndarray:
        """GPU-accelerated inverse FFT"""
        if self.use_gpu:
            data_gpu = cp.asarray(data)
            result = cp.fft.ifft(data_gpu, axis=axis)
            return cp.asnumpy(result)
        return np.fft.ifft(data, axis=axis)

    def rfft(self, data: np.ndarray, axis: int = -1) -> np.ndarray:
        """GPU-accelerated real FFT"""
        if self.use_gpu:
            data_gpu = cp.asarray(data)
            result = cp.fft.rfft(data_gpu, axis=axis)
            return cp.asnumpy(result)
        return np.fft.rfft(data, axis=axis)

    def hilbert(self, data: np.ndarray, axis: int = -1) -> np.ndarray:
        """GPU-accelerated Hilbert transform"""
        n = data.shape[axis]

        if self.use_gpu:
            data_gpu = cp.asarray(data)
            fft_data = cp.fft.fft(data_gpu, axis=axis)

            # Create Hilbert filter
            h = cp.zeros(n)
            if n % 2 == 0:
                h[0] = h[n//2] = 1
                h[1:n//2] = 2
            else:
                h[0] = 1
                h[1:(n+1)//2] = 2

            # Apply filter
            shape = [1] * data.ndim
            shape[axis] = n
            h = h.reshape(shape)

            result = cp.fft.ifft(fft_data * h, axis=axis)
            return cp.asnumpy(result)
        else:
            from scipy.signal import hilbert
            return hilbert(data, axis=axis)

    def correlate2d(self, data: np.ndarray, kernel: np.ndarray, mode: str = 'same') -> np.ndarray:
        """GPU-accelerated 2D correlation"""
        if self.use_gpu:
            from cupyx.scipy.ndimage import correlate
            data_gpu = cp.asarray(data)
            kernel_gpu = cp.asarray(kernel)
            result = correlate(data_gpu, kernel_gpu, mode='constant')
            return cp.asnumpy(result)
        else:
            from scipy.ndimage import correlate
            return correlate(data, kernel, mode='constant')

    def interpolate_traces(self, data: np.ndarray, dead_mask: np.ndarray,
                           method: str = 'linear') -> np.ndarray:
        """GPU-accelerated dead trace interpolation"""
        result = data.copy()

        if self.use_gpu:
            # Use GPU for the heavy computation
            data_gpu = cp.asarray(data)
            mask_gpu = cp.asarray(dead_mask)

            # For each dead trace, interpolate from neighbors
            dead_indices = cp.where(mask_gpu)[0]

            for idx in dead_indices:
                # Find nearest live traces
                left = idx - 1
                right = idx + 1

                while left >= 0 and mask_gpu[left]:
                    left -= 1
                while right < len(mask_gpu) and mask_gpu[right]:
                    right += 1

                # Interpolate
                if left >= 0 and right < len(mask_gpu):
                    weight = (idx - left) / (right - left)
                    data_gpu[idx] = (1 - weight) * data_gpu[left] + weight * data_gpu[right]
                elif left >= 0:
                    data_gpu[idx] = data_gpu[left]
                elif right < len(mask_gpu):
                    data_gpu[idx] = data_gpu[right]

            result = cp.asnumpy(data_gpu)
        else:
            # CPU fallback
            dead_indices = np.where(dead_mask)[0]
            for idx in dead_indices:
                left = idx - 1
                right = idx + 1

                while left >= 0 and dead_mask[left]:
                    left -= 1
                while right < len(dead_mask) and dead_mask[right]:
                    right += 1

                if left >= 0 and right < len(dead_mask):
                    weight = (idx - left) / (right - left)
                    result[idx] = (1 - weight) * data[left] + weight * data[right]
                elif left >= 0:
                    result[idx] = data[left]
                elif right < len(dead_mask):
                    result[idx] = data[right]

        return result

    def compute_envelope(self, data: np.ndarray, axis: int = -1) -> np.ndarray:
        """Compute instantaneous amplitude (envelope) using Hilbert transform"""
        analytic = self.hilbert(data, axis=axis)
        return np.abs(analytic)

    def compute_phase(self, data: np.ndarray, axis: int = -1) -> np.ndarray:
        """Compute instantaneous phase"""
        analytic = self.hilbert(data, axis=axis)
        return np.angle(analytic)

    def compute_frequency(self, data: np.ndarray, dt: float, axis: int = -1) -> np.ndarray:
        """Compute instantaneous frequency"""
        analytic = self.hilbert(data, axis=axis)
        phase = np.unwrap(np.angle(analytic), axis=axis)
        freq = np.gradient(phase, dt, axis=axis) / (2 * np.pi)
        return freq

    def bandpass_filter(self, data: np.ndarray, dt: float,
                        lowcut: float, highcut: float) -> np.ndarray:
        """GPU-accelerated bandpass filter"""
        n = data.shape[-1]
        freqs = np.fft.rfftfreq(n, dt)

        # Create filter
        filt = np.zeros(len(freqs))
        idx = (freqs >= lowcut) & (freqs <= highcut)
        filt[idx] = 1.0

        # Taper edges
        taper_width = 0.1 * (highcut - lowcut)
        low_taper = (freqs >= lowcut - taper_width) & (freqs < lowcut)
        high_taper = (freqs > highcut) & (freqs <= highcut + taper_width)
        filt[low_taper] = 0.5 * (1 + np.cos(np.pi * (lowcut - freqs[low_taper]) / taper_width))
        filt[high_taper] = 0.5 * (1 + np.cos(np.pi * (freqs[high_taper] - highcut) / taper_width))

        if self.use_gpu:
            data_gpu = cp.asarray(data)
            filt_gpu = cp.asarray(filt)

            fft_data = cp.fft.rfft(data_gpu, axis=-1)
            filtered = fft_data * filt_gpu
            result = cp.fft.irfft(filtered, n=n, axis=-1)
            return cp.asnumpy(result)
        else:
            fft_data = np.fft.rfft(data, axis=-1)
            filtered = fft_data * filt
            return np.fft.irfft(filtered, n=n, axis=-1)

    def compute_semblance(self, data: np.ndarray, window: int = 11) -> np.ndarray:
        """GPU-accelerated semblance (coherence) calculation"""
        if self.use_gpu:
            data_gpu = cp.asarray(data)

            # Sum of squares and square of sums
            sum_sq = cp.zeros_like(data_gpu)
            sq_sum = cp.zeros_like(data_gpu)

            half = window // 2
            for i in range(-half, half + 1):
                shifted = cp.roll(data_gpu, i, axis=0)
                sum_sq += shifted ** 2
                sq_sum += shifted

            sq_sum = sq_sum ** 2

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                semblance = sq_sum / (window * sum_sq + 1e-10)

            return cp.asnumpy(semblance)
        else:
            sum_sq = np.zeros_like(data)
            sq_sum = np.zeros_like(data)

            half = window // 2
            for i in range(-half, half + 1):
                shifted = np.roll(data, i, axis=0)
                sum_sq += shifted ** 2
                sq_sum += shifted

            sq_sum = sq_sum ** 2

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                semblance = sq_sum / (window * sum_sq + 1e-10)

            return semblance


# =============================================================================
# Convenience function
# =============================================================================

def create_gpu_ops(use_gpu: bool = True) -> GPUSeismicOps:
    """Create GPU operations handler"""
    return GPUSeismicOps(use_gpu=use_gpu)


# =============================================================================
# Test GPU
# =============================================================================

if __name__ == "__main__":
    print("GPU Utilities Test")
    print("=" * 50)

    info = gpu_info()
    print(f"GPU Available: {info['available']}")
    if info['available']:
        print(f"Device: {info['device_name']}")
        print(f"Memory: {info['memory_total_gb']:.1f} GB total, {info['memory_free_gb']:.1f} GB free")
        print(f"CuPy Version: {info['cupy_version']}")

    # Test operations
    print("\nTesting operations...")
    ops = create_gpu_ops(use_gpu=True)

    # Generate test data
    test_data = np.random.randn(1000, 2000).astype(np.float32)

    import time

    # FFT test
    start = time.time()
    _ = ops.rfft(test_data)
    print(f"FFT (1000x2000): {time.time()-start:.3f}s")

    # Hilbert test
    start = time.time()
    _ = ops.compute_envelope(test_data[0])
    print(f"Envelope (2000 samples): {time.time()-start:.3f}s")

    print("\nGPU utilities ready!")
