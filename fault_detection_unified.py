"""
================================================================================
UNIFIED FAULT DETECTION INTERFACE
================================================================================

Provides a common interface and data model for fault detection results from
both Deep Learning (FaultSeg3D) and Classical (Variance/Gradient) approaches.

This enables:
- Consistent output format across methods
- Easy comparison between methods
- Ensemble mode combining multiple approaches
- Single integration point for visualization

Author: Moses Ekene Obasi
PhD Research - University of Calabar
================================================================================
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
from datetime import datetime
import numpy as np
import json
from pathlib import Path


class FaultDetectionMethod(Enum):
    """Supported fault detection methods."""
    DEEP_LEARNING = "deep_learning"          # FaultSeg3D CNN
    CLASSICAL_VARIANCE = "classical_variance"  # Local variance attribute
    CLASSICAL_GRADIENT = "classical_gradient"  # Horizon gradient method
    ENSEMBLE = "ensemble"                      # Combined methods


@dataclass
class FaultSegment:
    """
    Individual fault segment with unified properties.

    This data structure can be populated by either deep learning
    or classical methods, providing a consistent interface.
    """
    fault_id: int
    method: FaultDetectionMethod

    # Spatial extent
    inline_range: Tuple[int, int]         # (min_il, max_il)
    crossline_range: Tuple[int, int]      # (min_xl, max_xl)
    time_range: Tuple[float, float]       # (min_twt, max_twt) in ms

    # Geometry
    strike_azimuth: float = 0.0           # Degrees from north (0-360)
    dip_angle: float = 90.0               # Degrees from horizontal (0-90)
    throw_estimate: Optional[float] = None  # Vertical displacement in ms
    length_km: Optional[float] = None     # Fault trace length

    # Confidence metrics
    confidence: float = 0.5               # 0-1, unified confidence score
    dl_probability: Optional[float] = None  # Deep learning probability (if available)
    classical_score: Optional[float] = None  # Classical attribute score (if available)

    # Trace data for visualization
    trace_points: List[Dict] = field(default_factory=list)  # [{"il": x, "xl": y, "twt": z}, ...]

    # 3D volume indices (for volumetric rendering)
    voxel_indices: Optional[np.ndarray] = None  # Shape (N, 3) array of [il, xl, sample] indices

    # Metadata
    interpreter_notes: str = ""
    validated: bool = False
    validation_well: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "fault_id": self.fault_id,
            "method": self.method.value,
            "inline_range": list(self.inline_range),
            "crossline_range": list(self.crossline_range),
            "time_range": list(self.time_range),
            "strike_azimuth": self.strike_azimuth,
            "dip_angle": self.dip_angle,
            "throw_estimate": self.throw_estimate,
            "length_km": self.length_km,
            "confidence": self.confidence,
            "dl_probability": self.dl_probability,
            "classical_score": self.classical_score,
            "trace_points": self.trace_points,
            "validated": self.validated,
            "validation_well": self.validation_well,
            "interpreter_notes": self.interpreter_notes
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'FaultSegment':
        """Create FaultSegment from dictionary."""
        return cls(
            fault_id=data["fault_id"],
            method=FaultDetectionMethod(data["method"]),
            inline_range=tuple(data["inline_range"]),
            crossline_range=tuple(data["crossline_range"]),
            time_range=tuple(data["time_range"]),
            strike_azimuth=data.get("strike_azimuth", 0.0),
            dip_angle=data.get("dip_angle", 90.0),
            throw_estimate=data.get("throw_estimate"),
            length_km=data.get("length_km"),
            confidence=data.get("confidence", 0.5),
            dl_probability=data.get("dl_probability"),
            classical_score=data.get("classical_score"),
            trace_points=data.get("trace_points", []),
            voxel_indices=None,
            validated=data.get("validated", False),
            validation_well=data.get("validation_well"),
            interpreter_notes=data.get("interpreter_notes", "")
        )


@dataclass
class UnifiedFaultResult:
    """
    Container for all fault detection results.

    Provides a unified format for results from any detection method,
    enabling consistent visualization and analysis.
    """
    # Identification
    survey_name: str
    detection_timestamp: str
    methods_used: List[FaultDetectionMethod]

    # Volume info
    volume_shape: Tuple[int, int, int]    # (n_inlines, n_crosslines, n_samples)
    inline_range: Tuple[int, int]
    crossline_range: Tuple[int, int]
    sample_rate_ms: float

    # Fault data
    faults: List[FaultSegment] = field(default_factory=list)

    # Full probability volume (from DL)
    probability_volume_path: Optional[str] = None

    # Statistics
    total_faults: int = 0
    major_faults: int = 0                 # Faults with length > threshold
    fault_volume_percent: float = 0.0

    # Confidence summary
    high_confidence_count: int = 0        # confidence > 0.8
    medium_confidence_count: int = 0      # 0.5 < confidence <= 0.8
    low_confidence_count: int = 0         # confidence <= 0.5

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "survey_name": self.survey_name,
            "detection_timestamp": self.detection_timestamp,
            "methods_used": [m.value for m in self.methods_used],
            "volume_shape": list(self.volume_shape),
            "inline_range": list(self.inline_range),
            "crossline_range": list(self.crossline_range),
            "sample_rate_ms": self.sample_rate_ms,
            "faults": [f.to_dict() for f in self.faults],
            "probability_volume_path": self.probability_volume_path,
            "statistics": {
                "total_faults": self.total_faults,
                "major_faults": self.major_faults,
                "fault_volume_percent": self.fault_volume_percent,
                "high_confidence_count": self.high_confidence_count,
                "medium_confidence_count": self.medium_confidence_count,
                "low_confidence_count": self.low_confidence_count
            }
        }

    def save(self, filepath: str):
        """Save results to JSON."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Saved unified fault results to: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'UnifiedFaultResult':
        """Load results from JSON."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        result = cls(
            survey_name=data["survey_name"],
            detection_timestamp=data["detection_timestamp"],
            methods_used=[FaultDetectionMethod(m) for m in data["methods_used"]],
            volume_shape=tuple(data["volume_shape"]),
            inline_range=tuple(data["inline_range"]),
            crossline_range=tuple(data["crossline_range"]),
            sample_rate_ms=data["sample_rate_ms"],
            probability_volume_path=data.get("probability_volume_path")
        )

        # Load faults
        for fd in data.get("faults", []):
            result.faults.append(FaultSegment.from_dict(fd))

        # Load stats
        stats = data.get("statistics", {})
        result.total_faults = stats.get("total_faults", len(result.faults))
        result.major_faults = stats.get("major_faults", 0)
        result.fault_volume_percent = stats.get("fault_volume_percent", 0.0)
        result.high_confidence_count = stats.get("high_confidence_count", 0)
        result.medium_confidence_count = stats.get("medium_confidence_count", 0)
        result.low_confidence_count = stats.get("low_confidence_count", 0)

        return result

    def get_faults_by_method(self, method: FaultDetectionMethod) -> List[FaultSegment]:
        """Get faults detected by a specific method."""
        return [f for f in self.faults if f.method == method]

    def get_high_confidence_faults(self, threshold: float = 0.8) -> List[FaultSegment]:
        """Get faults above confidence threshold."""
        return [f for f in self.faults if f.confidence >= threshold]

    def get_faults_in_region(self, il_range: Tuple[int, int],
                             xl_range: Tuple[int, int]) -> List[FaultSegment]:
        """Get faults within specified inline/crossline region."""
        result = []
        for f in self.faults:
            # Check if fault intersects region
            il_overlap = not (f.inline_range[1] < il_range[0] or
                             f.inline_range[0] > il_range[1])
            xl_overlap = not (f.crossline_range[1] < xl_range[0] or
                             f.crossline_range[0] > xl_range[1])
            if il_overlap and xl_overlap:
                result.append(f)
        return result


class UnifiedFaultDetector:
    """
    Unified interface for fault detection.

    Combines Deep Learning (primary) with Classical methods (validation),
    outputting results in a consistent format.

    Example:
        detector = UnifiedFaultDetector(seismic_data, geometry)
        results = detector.detect_faults(method=FaultDetectionMethod.DEEP_LEARNING)
        results.save("unified_faults.json")
    """

    def __init__(self, seismic_data: np.ndarray, geometry: Dict):
        """
        Initialize detector with seismic data.

        Args:
            seismic_data: 3D numpy array (inline, crossline, samples)
            geometry: Dict with keys: il_min, il_max, xl_min, xl_max,
                      sample_rate_ms, survey_name (optional)
        """
        self.seismic_data = seismic_data
        self.geometry = geometry
        self.results: Optional[UnifiedFaultResult] = None

        # Method availability
        self._dl_available = self._check_dl_available()
        self._scipy_available = self._check_scipy_available()

    def _check_dl_available(self) -> bool:
        """Check if deep learning dependencies are available."""
        try:
            import torch
            return True
        except ImportError:
            return False

    def _check_scipy_available(self) -> bool:
        """Check if scipy is available for classical methods."""
        try:
            from scipy import ndimage
            return True
        except ImportError:
            return False

    def detect_faults(self,
                      method: FaultDetectionMethod = FaultDetectionMethod.DEEP_LEARNING,
                      probability_threshold: float = 0.5,
                      min_fault_size: int = 100,
                      use_ensemble: bool = False,
                      output_dir: Optional[str] = None) -> UnifiedFaultResult:
        """
        Run fault detection with specified method.

        Args:
            method: Detection method to use
            probability_threshold: Threshold for DL probability (0-1)
            min_fault_size: Minimum fault size in voxels
            use_ensemble: If True, run both DL and classical, combine results
            output_dir: Directory to save probability volumes

        Returns:
            UnifiedFaultResult with detected faults
        """
        methods_used = []
        all_faults = []
        probability_volume_path = None

        # Run Deep Learning detection
        if method in [FaultDetectionMethod.DEEP_LEARNING, FaultDetectionMethod.ENSEMBLE]:
            if self._dl_available:
                dl_faults, prob_path = self._run_deep_learning(
                    probability_threshold, min_fault_size, output_dir
                )
                all_faults.extend(dl_faults)
                methods_used.append(FaultDetectionMethod.DEEP_LEARNING)
                probability_volume_path = prob_path
            else:
                print("Warning: Deep learning not available, falling back to classical")
                method = FaultDetectionMethod.CLASSICAL_VARIANCE

        # Run Classical detection (for validation or primary)
        if method in [FaultDetectionMethod.CLASSICAL_VARIANCE,
                      FaultDetectionMethod.CLASSICAL_GRADIENT,
                      FaultDetectionMethod.ENSEMBLE] or use_ensemble:
            if self._scipy_available:
                classical_faults = self._run_classical(probability_threshold)
                all_faults.extend(classical_faults)
                if FaultDetectionMethod.CLASSICAL_VARIANCE not in methods_used:
                    methods_used.append(FaultDetectionMethod.CLASSICAL_VARIANCE)

        # If ensemble, merge overlapping faults
        if use_ensemble and len(all_faults) > 0:
            all_faults = self._merge_ensemble_faults(all_faults)

        # Build result
        self.results = UnifiedFaultResult(
            survey_name=self.geometry.get("survey_name", "Unknown"),
            detection_timestamp=datetime.now().isoformat(),
            methods_used=methods_used,
            volume_shape=self.seismic_data.shape,
            inline_range=(self.geometry.get("il_min", 0),
                         self.geometry.get("il_max", self.seismic_data.shape[0]-1)),
            crossline_range=(self.geometry.get("xl_min", 0),
                            self.geometry.get("xl_max", self.seismic_data.shape[1]-1)),
            sample_rate_ms=self.geometry.get("sample_rate_ms", 4.0),
            faults=all_faults,
            probability_volume_path=probability_volume_path
        )

        # Calculate statistics
        self._calculate_statistics()

        return self.results

    def _run_deep_learning(self, threshold: float, min_size: int,
                           output_dir: Optional[str]) -> Tuple[List[FaultSegment], Optional[str]]:
        """Run FaultSeg3D deep learning detection."""
        try:
            from deep_learning.dl_fault_detection import FaultDetector, FaultDetectionConfig

            config = FaultDetectionConfig(
                probability_threshold=threshold,
                min_fault_size=min_size,
                output_dir=output_dir or "fault_outputs"
            )
            detector = FaultDetector(config)
            dl_results = detector.detect_faults(seismic_data=self.seismic_data)

            # Convert DL results to unified format
            faults = []
            for i, fault_data in enumerate(dl_results.fault_orientations):
                fault = FaultSegment(
                    fault_id=i,
                    method=FaultDetectionMethod.DEEP_LEARNING,
                    inline_range=self._extract_il_range(fault_data),
                    crossline_range=self._extract_xl_range(fault_data),
                    time_range=self._extract_time_range(fault_data),
                    strike_azimuth=fault_data.get("strike", 0.0),
                    dip_angle=fault_data.get("dip", 90.0),
                    throw_estimate=None,
                    length_km=None,
                    confidence=min(1.0, fault_data.get("mean_probability", 0.5) * 1.2),
                    dl_probability=fault_data.get("mean_probability"),
                    classical_score=None,
                    trace_points=self._extract_trace_points(fault_data)
                )
                faults.append(fault)

            return faults, dl_results.probability_file

        except Exception as e:
            print(f"Deep learning fault detection error: {e}")
            return [], None

    def _run_classical(self, threshold: float) -> List[FaultSegment]:
        """Run classical variance/gradient detection."""
        from scipy import ndimage

        faults = []

        # Calculate variance attribute on time slices
        n_il, n_xl, n_samples = self.seismic_data.shape
        variance_volume = np.zeros_like(self.seismic_data, dtype=np.float32)

        print("Computing variance attribute...")
        for t in range(n_samples):
            slice_data = self.seismic_data[:, :, t].astype(np.float32)
            # Local variance with 3x3 window
            mean_local = ndimage.uniform_filter(slice_data, size=3)
            sqr_mean = ndimage.uniform_filter(slice_data**2, size=3)
            variance_volume[:, :, t] = sqr_mean - mean_local**2

        # Normalize variance
        vmin, vmax = variance_volume.min(), variance_volume.max()
        if vmax > vmin:
            variance_volume = (variance_volume - vmin) / (vmax - vmin)

        # Threshold to get fault candidates
        fault_mask = variance_volume > threshold

        # Label connected components
        labeled, n_features = ndimage.label(fault_mask)

        print(f"Found {n_features} classical fault candidates")

        for fault_id in range(1, min(n_features + 1, 51)):  # Top 50
            coords = np.where(labeled == fault_id)
            if len(coords[0]) < 100:  # Skip small features
                continue

            sample_rate = self.geometry.get("sample_rate_ms", 4.0)

            fault = FaultSegment(
                fault_id=fault_id + 1000,  # Offset to distinguish from DL
                method=FaultDetectionMethod.CLASSICAL_VARIANCE,
                inline_range=(int(coords[0].min()), int(coords[0].max())),
                crossline_range=(int(coords[1].min()), int(coords[1].max())),
                time_range=(float(coords[2].min() * sample_rate),
                           float(coords[2].max() * sample_rate)),
                strike_azimuth=self._estimate_strike(coords),
                dip_angle=self._estimate_dip(coords),
                throw_estimate=None,
                length_km=None,
                confidence=0.6,  # Lower confidence for classical
                dl_probability=None,
                classical_score=float(variance_volume[coords].mean()),
                trace_points=self._coords_to_trace(coords)
            )
            faults.append(fault)

        return faults

    def _merge_ensemble_faults(self, faults: List[FaultSegment]) -> List[FaultSegment]:
        """Merge overlapping faults from different methods."""
        merged = []
        used = set()

        for i, f1 in enumerate(faults):
            if i in used:
                continue

            # Find overlapping faults
            overlapping = [f1]
            for j, f2 in enumerate(faults[i+1:], start=i+1):
                if j in used:
                    continue
                if self._faults_overlap(f1, f2):
                    overlapping.append(f2)
                    used.add(j)

            # Merge overlapping faults
            if len(overlapping) > 1:
                merged_fault = self._merge_fault_group(overlapping)
                merged.append(merged_fault)
            else:
                merged.append(f1)
            used.add(i)

        return merged

    def _faults_overlap(self, f1: FaultSegment, f2: FaultSegment) -> bool:
        """Check if two faults spatially overlap."""
        # Check inline overlap
        il_overlap = not (f1.inline_range[1] < f2.inline_range[0] or
                         f2.inline_range[1] < f1.inline_range[0])
        # Check crossline overlap
        xl_overlap = not (f1.crossline_range[1] < f2.crossline_range[0] or
                         f2.crossline_range[1] < f1.crossline_range[0])
        return il_overlap and xl_overlap

    def _merge_fault_group(self, faults: List[FaultSegment]) -> FaultSegment:
        """Merge a group of overlapping faults into one."""
        # Use highest confidence fault as base
        base = max(faults, key=lambda f: f.confidence)

        # Expand ranges
        il_min = min(f.inline_range[0] for f in faults)
        il_max = max(f.inline_range[1] for f in faults)
        xl_min = min(f.crossline_range[0] for f in faults)
        xl_max = max(f.crossline_range[1] for f in faults)
        t_min = min(f.time_range[0] for f in faults)
        t_max = max(f.time_range[1] for f in faults)

        # Average confidence (boost for ensemble agreement)
        avg_conf = sum(f.confidence for f in faults) / len(faults)
        boosted_conf = min(1.0, avg_conf * 1.2)  # 20% boost for agreement

        return FaultSegment(
            fault_id=base.fault_id,
            method=FaultDetectionMethod.ENSEMBLE,
            inline_range=(il_min, il_max),
            crossline_range=(xl_min, xl_max),
            time_range=(t_min, t_max),
            strike_azimuth=base.strike_azimuth,
            dip_angle=base.dip_angle,
            throw_estimate=base.throw_estimate,
            length_km=base.length_km,
            confidence=boosted_conf,
            dl_probability=next((f.dl_probability for f in faults if f.dl_probability), None),
            classical_score=next((f.classical_score for f in faults if f.classical_score), None),
            trace_points=base.trace_points
        )

    def _calculate_statistics(self):
        """Calculate summary statistics for results."""
        if not self.results:
            return

        self.results.total_faults = len(self.results.faults)
        self.results.high_confidence_count = sum(
            1 for f in self.results.faults if f.confidence > 0.8
        )
        self.results.medium_confidence_count = sum(
            1 for f in self.results.faults if 0.5 < f.confidence <= 0.8
        )
        self.results.low_confidence_count = sum(
            1 for f in self.results.faults if f.confidence <= 0.5
        )

    # Helper methods for coordinate extraction
    def _extract_il_range(self, data: Dict) -> Tuple[int, int]:
        centroid = data.get("centroid", [0, 0, 0])
        size = data.get("size_voxels", 100)
        extent = int(np.sqrt(size) / 2)
        return (max(0, int(centroid[0]) - extent),
                min(self.seismic_data.shape[0]-1, int(centroid[0]) + extent))

    def _extract_xl_range(self, data: Dict) -> Tuple[int, int]:
        centroid = data.get("centroid", [0, 0, 0])
        size = data.get("size_voxels", 100)
        extent = int(np.sqrt(size) / 2)
        return (max(0, int(centroid[1]) - extent),
                min(self.seismic_data.shape[1]-1, int(centroid[1]) + extent))

    def _extract_time_range(self, data: Dict) -> Tuple[float, float]:
        centroid = data.get("centroid", [0, 0, 0])
        sr = self.geometry.get("sample_rate_ms", 4.0)
        return (max(0, centroid[2] * sr - 50), centroid[2] * sr + 50)

    def _extract_trace_points(self, data: Dict) -> List[Dict]:
        centroid = data.get("centroid", [0, 0, 0])
        sr = self.geometry.get("sample_rate_ms", 4.0)
        return [{"il": int(centroid[0]), "xl": int(centroid[1]),
                 "twt": float(centroid[2] * sr)}]

    def _estimate_strike(self, coords) -> float:
        if len(coords[0]) < 2:
            return 0.0
        dx = float(coords[0].max() - coords[0].min())
        dy = float(coords[1].max() - coords[1].min())
        return float(np.degrees(np.arctan2(dy, dx)))

    def _estimate_dip(self, coords) -> float:
        if len(coords[0]) < 2:
            return 90.0
        horiz = np.sqrt((coords[0].max() - coords[0].min())**2 +
                       (coords[1].max() - coords[1].min())**2)
        vert = coords[2].max() - coords[2].min()
        if horiz == 0:
            return 90.0
        return float(np.degrees(np.arctan2(vert, horiz)))

    def _coords_to_trace(self, coords) -> List[Dict]:
        """Convert coordinates to trace points."""
        sr = self.geometry.get("sample_rate_ms", 4.0)
        points = []
        # Sample every 10th point
        step = max(1, len(coords[0]) // 100)
        for i in range(0, len(coords[0]), step):
            points.append({
                "il": int(coords[0][i]),
                "xl": int(coords[1][i]),
                "twt": float(coords[2][i] * sr)
            })
        return points[:100]  # Limit to 100 points


def convert_dl_results_to_unified(dl_results, geometry: Dict,
                                   survey_name: str = "Unknown") -> UnifiedFaultResult:
    """
    Convert deep learning fault detection results to unified format.

    Args:
        dl_results: FaultDetectionResults from dl_fault_detection.py
        geometry: Dict with survey geometry
        survey_name: Name of the survey

    Returns:
        UnifiedFaultResult
    """
    faults = []

    for i, fault_data in enumerate(dl_results.fault_orientations):
        centroid = fault_data.get("centroid", [0, 0, 0])
        size = fault_data.get("size_voxels", 100)
        extent = int(np.sqrt(size) / 2)
        sr = geometry.get("sample_rate_ms", 4.0)

        fault = FaultSegment(
            fault_id=i,
            method=FaultDetectionMethod.DEEP_LEARNING,
            inline_range=(int(centroid[0]) - extent, int(centroid[0]) + extent),
            crossline_range=(int(centroid[1]) - extent, int(centroid[1]) + extent),
            time_range=(centroid[2] * sr - 50, centroid[2] * sr + 50),
            strike_azimuth=fault_data.get("strike", 0.0),
            dip_angle=fault_data.get("dip", 90.0),
            confidence=min(1.0, fault_data.get("mean_probability", 0.5) * 1.2),
            dl_probability=fault_data.get("mean_probability"),
            trace_points=[{"il": int(centroid[0]), "xl": int(centroid[1]),
                          "twt": float(centroid[2] * sr)}]
        )
        faults.append(fault)

    return UnifiedFaultResult(
        survey_name=survey_name,
        detection_timestamp=datetime.now().isoformat(),
        methods_used=[FaultDetectionMethod.DEEP_LEARNING],
        volume_shape=(0, 0, 0),  # Fill in from actual data
        inline_range=(geometry.get("il_min", 0), geometry.get("il_max", 0)),
        crossline_range=(geometry.get("xl_min", 0), geometry.get("xl_max", 0)),
        sample_rate_ms=geometry.get("sample_rate_ms", 4.0),
        faults=faults,
        probability_volume_path=dl_results.probability_file,
        total_faults=len(faults),
        fault_volume_percent=dl_results.fault_volume_percent
    )


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Unified Fault Detection")
    parser.add_argument("--convert", type=str,
                        help="Convert DL results JSON to unified format")
    parser.add_argument("--output", type=str, default="unified_faults.json",
                        help="Output file path")

    args = parser.parse_args()

    if args.convert:
        # Load DL results and convert
        with open(args.convert, 'r') as f:
            dl_data = json.load(f)

        # Create mock DL results object
        class MockResults:
            pass

        results = MockResults()
        results.fault_orientations = dl_data.get("fault_orientations", [])
        results.probability_file = dl_data.get("probability_file", "")
        results.fault_volume_percent = dl_data.get("fault_volume_percent", 0.0)

        geometry = {
            "il_min": 0, "il_max": 1000,
            "xl_min": 0, "xl_max": 1000,
            "sample_rate_ms": 4.0
        }

        unified = convert_dl_results_to_unified(results, geometry)
        unified.save(args.output)
        print(f"Converted to unified format: {args.output}")
