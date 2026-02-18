"""
===============================================================================
WELL DATA INTEGRATION AUTOMATION FRAMEWORK v5.0
Production-Ready Well Log Analysis and Velocity Modeling
===============================================================================

Author: Moses Ekene Obasi
Institution: University of Calabar, Nigeria
Supervisor: Prof. Dominic Akam Obi

Features:
- Multi-well LAS file processing
- Automated quality assessment and ranking
- Petrophysical calculations (Vsh, porosity, Sw, permeability)
- Formation identification
- Velocity model building (Gaussian Process)
- Time-depth conversion
- JSON structured output for automation
- CLI, API, and webhook support

Petrophysical Methods:
- Vsh: Larionov (1969) for Cretaceous rocks
- Porosity: Density-Sonic average with Raymer-Hunt
- Sw: Archie (1942) water saturation
- K: Timur (1968) permeability
- Velocity: Gardner (1974) relation

Usage:
    python well_integration_automation.py "path/to/las_folder" -o "output_dir"
    python well_integration_automation.py -c config.json
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lasio
from scipy import interpolate, stats
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import argparse
import hashlib
import logging
import pickle
import json
import os
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class WellIntegrationConfig:
    """Configuration for well data integration"""

    # Input/Output
    las_directory: str = ""
    output_dir: str = "well_outputs"

    # Well files (if empty, scans directory for .las files)
    well_files: Dict[str, str] = field(default_factory=dict)

    # Quality thresholds
    min_coverage_pct: float = 30.0
    good_coverage_pct: float = 70.0
    excellent_coverage_pct: float = 90.0

    # Petrophysical parameters
    rho_matrix: float = 2.65  # Sandstone g/cm3
    rho_fluid: float = 1.0
    archie_a: float = 1.0
    archie_m: float = 2.0
    archie_n: float = 2.0
    gr_clean: float = 30.0  # API
    gr_shale: float = 148.0  # API
    dt_matrix: float = 182.0  # us/m
    dt_fluid: float = 620.0  # us/m
    rw_default: float = 0.3  # ohm-m

    # Formation definitions (depth_top_ms, depth_bottom_ms, color)
    # Use generic names for publications - configure actual names in config file
    formations: Dict[str, Tuple[float, float, str]] = field(default_factory=lambda: {
        "Formation_A": (0, 1000, "#90EE90"),
        "Formation_B": (1000, 2500, "#696969"),
        "Formation_C": (2500, 3500, "#8B4513"),
        "Formation_D": (3500, 5000, "#FFD700")
    })

    # Velocity modeling
    build_velocity_model: bool = True
    velocity_model_type: str = "gp"  # gp (Gaussian Process) or rf (Random Forest)
    cross_validation: bool = True
    max_training_samples: int = 5000

    # Output options
    save_figures: bool = True
    figure_dpi: int = 300
    export_csv: bool = True
    export_velocity_functions: bool = True

    # Automation
    webhook_url: Optional[str] = None
    webhook_auth: Optional[str] = None
    progress_interval: int = 10

    # Caching
    enable_cache: bool = True
    cache_dir: str = ".well_cache"

    @classmethod
    def from_json(cls, path: str) -> 'WellIntegrationConfig':
        """Load configuration from JSON file, ignoring unknown fields"""
        with open(path, 'r') as f:
            data = json.load(f)
        # Handle formations dict if present and valid
        if 'formations' in data and isinstance(data['formations'], dict):
            data['formations'] = {k: tuple(v) for k, v in data['formations'].items()}
        # Only use fields that exist in this dataclass
        valid_fields = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid_fields)

    def to_json(self, path: str):
        """Save configuration to JSON file"""
        data = asdict(self)
        if 'formations' in data:
            data['formations'] = {k: list(v) for k, v in data['formations'].items()}
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)


@dataclass
class WellResults:
    """Structured results from well integration"""

    # Metadata
    timestamp: str = ""
    version: str = "5.0"
    processing_time_seconds: float = 0.0
    las_directory: str = ""
    basin: str = "Study Area"  # Configurable - use generic name for publications

    # Well quality assessment
    quality_assessment: Dict[str, Any] = field(default_factory=dict)
    tier_summary: Dict[str, int] = field(default_factory=dict)

    # Processed wells
    wells_processed: List[str] = field(default_factory=list)
    wells_failed: List[str] = field(default_factory=list)

    # Petrophysical summaries
    petrophysical_summary: Dict[str, Any] = field(default_factory=dict)

    # Velocity model
    velocity_model: Dict[str, Any] = field(default_factory=dict)

    # Reservoir summary
    reservoir_summary: List[Dict[str, Any]] = field(default_factory=list)

    # Quality
    quality: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Output files
    output_files: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


# =============================================================================
# PROGRESS TRACKING
# =============================================================================

class ProgressTracker:
    """Track and report progress with optional webhook callbacks"""

    def __init__(self, total_steps: int, webhook_url: Optional[str] = None,
                 webhook_auth: Optional[str] = None):
        self.total_steps = total_steps
        self.current_step = 0
        self.webhook_url = webhook_url
        self.webhook_auth = webhook_auth
        self.start_time = datetime.now()

    def update(self, step_name: str, progress: float, details: str = ""):
        """Update progress and optionally send webhook"""
        self.current_step += 1

        message = {
            "step": self.current_step,
            "total_steps": self.total_steps,
            "step_name": step_name,
            "progress_percent": progress,
            "details": details,
            "elapsed_seconds": (datetime.now() - self.start_time).total_seconds()
        }

        logger.info(f"[{self.current_step}/{self.total_steps}] {step_name}: {progress:.1f}% - {details}")

        if self.webhook_url:
            self._send_webhook(message)

    def _send_webhook(self, data: dict):
        """Send progress update via webhook"""
        try:
            import requests
            headers = {"Content-Type": "application/json"}
            if self.webhook_auth:
                headers["Authorization"] = f"Bearer {self.webhook_auth}"
            requests.post(self.webhook_url, json=data, headers=headers, timeout=5)
        except Exception as e:
            logger.warning(f"Webhook failed: {e}")


# =============================================================================
# PETROPHYSICS MODULE
# =============================================================================

class Petrophysics:
    """Literature-backed petrophysical calculations"""

    def __init__(self, config: WellIntegrationConfig):
        self.config = config

    def calculate_vshale_larionov(self, gr: np.ndarray) -> np.ndarray:
        """
        Larionov (1969) Vshale for Cretaceous rocks.
        Vsh = 0.33 * (2^(2*IGR) - 1)
        """
        igr = np.clip(
            (gr - self.config.gr_clean) / (self.config.gr_shale - self.config.gr_clean),
            0, 1
        )
        vsh = 0.33 * (2 ** (2 * igr) - 1)
        return np.clip(vsh, 0, 1)

    def calculate_porosity_density(self, rhob: np.ndarray) -> np.ndarray:
        """Density porosity (Schlumberger 1987)"""
        phi = (self.config.rho_matrix - rhob) / (self.config.rho_matrix - self.config.rho_fluid)
        return np.clip(phi, 0, 0.5)

    def calculate_porosity_sonic(self, dt: np.ndarray) -> np.ndarray:
        """Wyllie time-average with Raymer-Hunt correction"""
        phi = (dt - self.config.dt_matrix) / (self.config.dt_fluid - self.config.dt_matrix)
        return np.clip(phi * 0.625, 0, 0.45)  # Raymer correction

    def calculate_effective_porosity(self, phi_total: np.ndarray, vsh: np.ndarray) -> np.ndarray:
        """Effective porosity (Ezeobi et al. 2023)"""
        return np.clip(phi_total * (1 - vsh), 0, 0.5)

    def calculate_water_saturation(self, rt: np.ndarray, rw: float,
                                    porosity: np.ndarray) -> np.ndarray:
        """Archie (1942) water saturation"""
        a, m, n = self.config.archie_a, self.config.archie_m, self.config.archie_n
        sw = ((a * rw) / (porosity ** m * rt + 1e-10)) ** (1 / n)
        return np.clip(sw, 0, 1)

    def calculate_permeability(self, porosity: np.ndarray, swirr: np.ndarray) -> np.ndarray:
        """Timur (1968) permeability in mD"""
        return 0.136 * (porosity ** 4.4 / (swirr ** 2 + 1e-10))

    def gardners_relation(self, velocity: np.ndarray) -> np.ndarray:
        """Gardner et al. (1974) velocity-density relation"""
        return 0.23 * ((velocity / 1000) ** 0.25)


# =============================================================================
# MAIN AUTOMATION CLASS
# =============================================================================

class WellIntegrationAutomation:
    """
    Production-ready well data integration automation.
    """

    def __init__(self, config: WellIntegrationConfig):
        self.config = config
        self.results = WellResults()
        self.petro = Petrophysics(config)

        # Data storage
        self.well_quality = {}
        self.well_data = {}
        self.velocity_data = {}

        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    def _scan_las_files(self) -> Dict[str, str]:
        """Scan directory for LAS files"""
        las_files = {}
        las_dir = Path(self.config.las_directory)

        for las_file in las_dir.glob("*.las"):
            well_name = las_file.stem.replace("_", "-").upper()
            las_files[well_name] = las_file.name

        for las_file in las_dir.glob("*.LAS"):
            well_name = las_file.stem.replace("_", "-").upper()
            las_files[well_name] = las_file.name

        return las_files

    def _extract_log(self, las, log_names: List[str]) -> Optional[np.ndarray]:
        """Extract log from LAS file trying multiple name variations"""
        for name in log_names:
            if name in las.keys():
                data = np.array(las[name])
                null_value = las.well.NULL.value if hasattr(las.well, 'NULL') else -999.25
                data = np.where(data == null_value, np.nan, data)
                return data
        return None

    def _assess_log_quality(self, log_data: Optional[np.ndarray]) -> Dict[str, Any]:
        """Calculate coverage and quality metrics for a log"""
        if log_data is None:
            return {
                'present': False,
                'coverage_pct': 0,
                'min': None,
                'max': None,
                'mean': None,
                'quality': 'MISSING'
            }

        valid = ~np.isnan(log_data)
        coverage = np.sum(valid) / len(log_data) * 100

        if coverage < self.config.min_coverage_pct:
            quality = 'POOR'
        elif coverage < self.config.good_coverage_pct:
            quality = 'FAIR'
        elif coverage < self.config.excellent_coverage_pct:
            quality = 'GOOD'
        else:
            quality = 'EXCELLENT'

        return {
            'present': True,
            'coverage_pct': float(coverage),
            'min': float(np.nanmin(log_data)) if coverage > 0 else None,
            'max': float(np.nanmax(log_data)) if coverage > 0 else None,
            'mean': float(np.nanmean(log_data)) if coverage > 0 else None,
            'quality': quality
        }

    # =========================================================================
    # QUALITY ASSESSMENT
    # =========================================================================

    def assess_quality(self) -> Dict[str, Any]:
        """Assess quality of all wells"""
        logger.info("Assessing well quality...")

        well_files = self.config.well_files or self._scan_las_files()

        if not well_files:
            logger.warning("No LAS files found")
            return {}

        logger.info(f"Found {len(well_files)} LAS files")

        quality_results = {}

        for well_name, las_file in tqdm(well_files.items(), desc="Assessing wells"):
            las_path = Path(self.config.las_directory) / las_file

            if not las_path.exists():
                quality_results[well_name] = {
                    'status': 'FILE_NOT_FOUND',
                    'tier': 'EXCLUDED',
                    'score': 0
                }
                continue

            try:
                las = lasio.read(str(las_path))

                # Get depth
                if hasattr(las, 'depth_m'):
                    depth = np.array(las.depth_m)
                elif 'DEPT' in las.keys():
                    depth = np.array(las['DEPT'])
                elif 'DEPTH' in las.keys():
                    depth = np.array(las['DEPTH'])
                else:
                    depth = np.array(las.index)

                # Extract logs
                gr = self._extract_log(las, ['GR', 'GR_EDTC', 'CGR', 'GRC'])
                dt = self._extract_log(las, ['DT', 'DT8', 'DTCO', 'AC', 'DTC'])
                rhob = self._extract_log(las, ['RHOB', 'ZDEN', 'DEN', 'RHOZ', 'DENB'])
                res = self._extract_log(las, ['ILD', 'RILD', 'LLD', 'RT', 'RESD', 'LLS'])

                # Assess quality
                gr_q = self._assess_log_quality(gr)
                dt_q = self._assess_log_quality(dt)
                rhob_q = self._assess_log_quality(rhob)
                res_q = self._assess_log_quality(res)

                # Calculate score
                score = 0
                if dt_q['coverage_pct'] > 70:
                    score += 40
                elif dt_q['coverage_pct'] > 30:
                    score += 20

                if gr_q['coverage_pct'] > 70:
                    score += 30
                elif gr_q['coverage_pct'] > 30:
                    score += 15

                if rhob_q['coverage_pct'] > 70:
                    score += 20
                elif rhob_q['coverage_pct'] > 30:
                    score += 10

                if res_q['coverage_pct'] > 70:
                    score += 10
                elif res_q['coverage_pct'] > 30:
                    score += 5

                # Assign tier
                if score >= 80:
                    tier = 'TIER_1_EXCELLENT'
                elif score >= 60:
                    tier = 'TIER_2_GOOD'
                elif score >= 40:
                    tier = 'TIER_3_USABLE'
                else:
                    tier = 'EXCLUDED'

                quality_results[well_name] = {
                    'status': 'ANALYZED',
                    'file': las_file,
                    'depth_min': float(np.nanmin(depth)),
                    'depth_max': float(np.nanmax(depth)),
                    'depth_range': float(np.nanmax(depth) - np.nanmin(depth)),
                    'n_samples': len(depth),
                    'gr': gr_q,
                    'dt': dt_q,
                    'rhob': rhob_q,
                    'res': res_q,
                    'score': score,
                    'tier': tier
                }

            except Exception as e:
                logger.warning(f"Error processing {well_name}: {e}")
                quality_results[well_name] = {
                    'status': 'ERROR',
                    'error': str(e),
                    'tier': 'EXCLUDED',
                    'score': 0
                }

        self.well_quality = quality_results
        self.results.quality_assessment = quality_results

        # Tier summary
        tier_counts = {}
        for well, data in quality_results.items():
            tier = data.get('tier', 'EXCLUDED')
            tier_counts[tier] = tier_counts.get(tier, 0) + 1

        self.results.tier_summary = tier_counts

        logger.info(f"Quality assessment complete. Tier breakdown: {tier_counts}")

        return quality_results

    # =========================================================================
    # WELL PROCESSING
    # =========================================================================

    def process_wells(self, min_tier: str = "TIER_3_USABLE") -> int:
        """Process wells that meet minimum tier requirement"""
        logger.info(f"Processing wells (min tier: {min_tier})...")

        tier_order = ['TIER_1_EXCELLENT', 'TIER_2_GOOD', 'TIER_3_USABLE', 'EXCLUDED']
        min_tier_idx = tier_order.index(min_tier)

        wells_to_process = [
            well for well, data in self.well_quality.items()
            if tier_order.index(data.get('tier', 'EXCLUDED')) <= min_tier_idx
        ]

        logger.info(f"Processing {len(wells_to_process)} wells")

        for well_name in tqdm(wells_to_process, desc="Processing wells"):
            try:
                self._process_single_well(well_name)
                self.results.wells_processed.append(well_name)
            except Exception as e:
                logger.warning(f"Failed to process {well_name}: {e}")
                self.results.wells_failed.append(well_name)

        logger.info(f"Processed {len(self.results.wells_processed)} wells successfully")
        return len(self.results.wells_processed)

    def _process_single_well(self, well_name: str):
        """Process a single well with full petrophysics"""
        quality_data = self.well_quality.get(well_name, {})
        las_file = quality_data.get('file')

        if not las_file:
            raise ValueError(f"No file found for {well_name}")

        las_path = Path(self.config.las_directory) / las_file
        las = lasio.read(str(las_path))

        # Get depth
        if hasattr(las, 'depth_m'):
            depth = np.array(las.depth_m)
        elif 'DEPT' in las.keys():
            depth = np.array(las['DEPT'])
        else:
            depth = np.array(las.index)

        # Extract logs
        gr = self._extract_log(las, ['GR', 'GR_EDTC', 'CGR', 'GRC'])
        dt = self._extract_log(las, ['DT', 'DT8', 'DTCO', 'AC', 'DTC'])
        rhob = self._extract_log(las, ['RHOB', 'ZDEN', 'DEN', 'RHOZ', 'DENB'])
        res = self._extract_log(las, ['ILD', 'RILD', 'LLD', 'RT', 'RESD', 'LLS'])

        # Create validity mask
        if dt is not None and gr is not None:
            mask = ~np.isnan(dt) & ~np.isnan(gr)
            if rhob is not None:
                mask = mask & ~np.isnan(rhob)
        else:
            raise ValueError("Missing critical logs (DT or GR)")

        # Apply mask
        depth_clean = depth[mask]
        gr_clean = gr[mask]
        dt_clean = dt[mask]
        rhob_clean = rhob[mask] if rhob is not None else None
        res_clean = res[mask] if res is not None else None

        if len(depth_clean) < 10:
            raise ValueError(f"Insufficient valid data ({len(depth_clean)} samples)")

        # Calculate velocity from sonic transit time
        # DT is in microseconds per foot (µs/ft) from LAS files
        # Velocity (m/s) = 304800 / DT(µs/ft)
        # where 304800 = 1e6 µs/s × 0.3048 m/ft
        velocity = 304800 / dt_clean

        # Clip to physically reasonable range for sedimentary rocks:
        # 1500 m/s (unconsolidated) to 6500 m/s (tight limestone)
        velocity = np.clip(velocity, 1500, 6500)

        # Petrophysical calculations
        vsh = self.petro.calculate_vshale_larionov(gr_clean)

        if rhob_clean is not None:
            phi_d = self.petro.calculate_porosity_density(rhob_clean)
            phi_s = self.petro.calculate_porosity_sonic(dt_clean)
            phi_total = (phi_d + phi_s) / 2
        else:
            phi_total = self.petro.calculate_porosity_sonic(dt_clean)
            rhob_clean = self.petro.gardners_relation(velocity)

        phi_eff = self.petro.calculate_effective_porosity(phi_total, vsh)

        # Water saturation
        if res_clean is not None:
            clean_mask = (vsh < 0.1) & (res_clean < 10) & (res_clean > 0)
            if np.sum(clean_mask) > 10:
                F = 1.0 / (phi_eff[clean_mask] ** 2.0 + 1e-10)
                rw = np.median(res_clean[clean_mask] / F)
            else:
                rw = self.config.rw_default

            sw = self.petro.calculate_water_saturation(res_clean, rw, phi_eff)
            sh = 1.0 - sw

            swirr = np.clip((1.0 / (phi_eff ** 2.0 + 1e-10) / 2000) ** 0.5, 0.1, 0.5)
            perm = self.petro.calculate_permeability(phi_eff, swirr)
        else:
            sw, sh, perm = None, None, None

        # Calculate TWT
        twt = self._calculate_twt(depth_clean, velocity)

        # Store data
        self.well_data[well_name] = {
            'depth': depth_clean,
            'gr': gr_clean,
            'dt': dt_clean,
            'rhob': rhob_clean,
            'res': res_clean,
            'velocity': velocity,
            'vsh': vsh,
            'phi_total': phi_total,
            'phi_eff': phi_eff,
            'sw': sw,
            'sh': sh,
            'perm': perm,
            'twt': twt
        }

        self.velocity_data[well_name] = {
            'depth': depth_clean,
            'velocity': velocity,
            'twt': twt
        }

        logger.debug(f"  {well_name}: {len(depth_clean)} samples, velocity: {velocity.min():.0f}-{velocity.max():.0f} m/s")

    def _calculate_twt(self, depth: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        """Calculate two-way travel time"""
        twt = np.zeros(len(depth))
        for i in range(1, len(depth)):
            dz = depth[i] - depth[i - 1]
            v_avg = (velocity[i] + velocity[i - 1]) / 2
            dt = dz / v_avg
            twt[i] = twt[i - 1] + 2 * dt
        return twt * 1000  # Convert to ms

    # =========================================================================
    # VELOCITY MODEL
    # =========================================================================

    def build_velocity_model(self) -> Dict[str, Any]:
        """Build regional velocity model using Gaussian Process"""
        if len(self.velocity_data) < 3:
            logger.warning("Need at least 3 wells for velocity modeling")
            return {}

        logger.info("Building velocity model...")

        # Prepare training data
        X_train, y_train = [], []

        for well_name, data in self.velocity_data.items():
            # Subsample for efficiency
            step = max(1, len(data['depth']) // 100)
            for i in range(0, len(data['depth']), step):
                X_train.append([data['depth'][i]])
                y_train.append(data['velocity'][i])

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # Limit samples
        if len(X_train) > self.config.max_training_samples:
            indices = np.random.choice(len(X_train), self.config.max_training_samples, replace=False)
            X_train = X_train[indices]
            y_train = y_train[indices]

        logger.info(f"Training with {len(X_train)} samples")

        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, ConstantKernel
            from sklearn.preprocessing import StandardScaler

            # Normalize
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            X_scaled = scaler_X.fit_transform(X_train)
            y_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

            # Build GP
            kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=0.1)

            logger.info("Training Gaussian Process...")
            gp.fit(X_scaled, y_scaled)

            # Cross-validation
            if self.config.cross_validation:
                from sklearn.model_selection import cross_val_score
                scores = cross_val_score(gp, X_scaled, y_scaled, cv=5, scoring='r2')
                r2 = np.mean(scores)

                # RMSE
                y_pred = gp.predict(X_scaled)
                y_pred_inv = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()
                rmse = np.sqrt(np.mean((y_train - y_pred_inv) ** 2))
            else:
                r2 = None
                rmse = None

            model_result = {
                'type': 'gaussian_process',
                'kernel': str(gp.kernel_),
                'n_samples': len(X_train),
                'rmse': float(rmse) if rmse else None,
                'r2': float(r2) if r2 else None,
                'depth_range': [float(X_train.min()), float(X_train.max())],
                'velocity_range': [float(y_train.min()), float(y_train.max())]
            }

            # Save model
            model_path = Path(self.config.output_dir) / "velocity_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': gp,
                    'scaler_X': scaler_X,
                    'scaler_y': scaler_y
                }, f)

            model_result['model_file'] = str(model_path)

            self.results.velocity_model = model_result

            logger.info(f"Velocity model complete. R2: {r2:.3f}, RMSE: {rmse:.1f} m/s")

            return model_result

        except ImportError:
            logger.warning("sklearn not available, skipping velocity model")
            return {}

    # =========================================================================
    # RESERVOIR SUMMARY
    # =========================================================================

    def generate_reservoir_summary(self) -> List[Dict[str, Any]]:
        """Generate reservoir summary for all wells"""
        logger.info("Generating reservoir summary...")

        summaries = []

        for well_name, data in self.well_data.items():
            if data['sh'] is None:
                continue

            # Reservoir criteria: Sh > 50%, Vsh < 30%
            reservoir_mask = (data['sh'] > 0.5) & (data['vsh'] < 0.3) & (data['phi_eff'] > 0.05)

            if np.sum(reservoir_mask) < 10:
                continue

            rd = data['depth'][reservoir_mask]

            summary = {
                'well': well_name,
                'top_m': float(rd.min()),
                'bottom_m': float(rd.max()),
                'thickness_m': float(rd.max() - rd.min()),
                'net_pay_m': float(np.sum(reservoir_mask) * np.mean(np.diff(data['depth']))),
                'avg_vsh': float(data['vsh'][reservoir_mask].mean()),
                'avg_porosity_pct': float(data['phi_eff'][reservoir_mask].mean() * 100),
                'avg_sw_pct': float(data['sw'][reservoir_mask].mean() * 100),
                'avg_sh_pct': float(data['sh'][reservoir_mask].mean() * 100),
                'avg_perm_md': float(data['perm'][reservoir_mask].mean()) if data['perm'] is not None else None
            }

            summaries.append(summary)

        self.results.reservoir_summary = summaries

        if summaries:
            logger.info(f"Found reservoir intervals in {len(summaries)} wells")

        return summaries

    # =========================================================================
    # VISUALIZATION
    # =========================================================================

    def create_figures(self):
        """Generate publication-quality figures"""
        logger.info("Generating figures...")

        fig_dir = Path(self.config.output_dir) / "figures"
        fig_dir.mkdir(exist_ok=True)

        # Figure 1: Well quality ranking
        self._plot_quality_ranking(fig_dir)

        # Figure 2: Velocity-depth profiles
        self._plot_velocity_depth(fig_dir)

        # Figure 3: Log coverage heatmap
        self._plot_log_coverage(fig_dir)

        self.results.output_files['figures'] = str(fig_dir)

        logger.info(f"Figures saved to {fig_dir}")

    def _plot_quality_ranking(self, fig_dir: Path):
        """Plot well quality ranking"""
        if not self.well_quality:
            return

        fig, ax = plt.subplots(figsize=(12, 8))

        wells = []
        scores = []
        colors = []

        for well, data in sorted(self.well_quality.items(),
                                  key=lambda x: x[1].get('score', 0), reverse=True):
            if data.get('status') == 'ANALYZED':
                wells.append(well)
                scores.append(data['score'])
                tier = data.get('tier', 'EXCLUDED')
                if 'TIER_1' in tier:
                    colors.append('green')
                elif 'TIER_2' in tier:
                    colors.append('blue')
                elif 'TIER_3' in tier:
                    colors.append('orange')
                else:
                    colors.append('red')

        ax.barh(wells, scores, color=colors)
        ax.axvline(80, color='green', ls='--', alpha=0.5, label='Tier 1')
        ax.axvline(60, color='blue', ls='--', alpha=0.5, label='Tier 2')
        ax.axvline(40, color='orange', ls='--', alpha=0.5, label='Tier 3')
        ax.set_xlabel('Quality Score', fontweight='bold')
        ax.set_title('Well Quality Ranking', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3, axis='x')

        plt.tight_layout()
        fig.savefig(fig_dir / 'quality_ranking.png', dpi=self.config.figure_dpi, bbox_inches='tight')
        plt.close()

    def _plot_velocity_depth(self, fig_dir: Path):
        """Plot velocity-depth profiles"""
        if not self.velocity_data:
            return

        fig, ax = plt.subplots(figsize=(12, 10))

        colors = plt.cm.tab10(np.linspace(0, 1, len(self.velocity_data)))

        for i, (well_name, data) in enumerate(self.velocity_data.items()):
            ax.plot(data['velocity'], data['depth'], label=well_name,
                   linewidth=2, color=colors[i], alpha=0.8)

        ax.set_xlabel('Interval Velocity (m/s)', fontweight='bold')
        ax.set_ylabel('Depth (m)', fontweight='bold')
        ax.set_title('Velocity-Depth Profiles', fontweight='bold')
        ax.invert_yaxis()
        ax.grid(alpha=0.3)
        ax.legend(loc='best', ncol=2)
        ax.set_xlim(1500, 5500)

        plt.tight_layout()
        fig.savefig(fig_dir / 'velocity_depth.png', dpi=self.config.figure_dpi, bbox_inches='tight')
        plt.close()

    def _plot_log_coverage(self, fig_dir: Path):
        """Plot log coverage heatmap"""
        if not self.well_quality:
            return

        wells = []
        coverages = []

        for well, data in self.well_quality.items():
            if data.get('status') == 'ANALYZED':
                wells.append(well)
                coverages.append([
                    data['gr']['coverage_pct'],
                    data['dt']['coverage_pct'],
                    data['rhob']['coverage_pct'],
                    data['res']['coverage_pct']
                ])

        if not wells:
            return

        fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.imshow(coverages, aspect='auto', cmap='RdYlGn', vmin=0, vmax=100)
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(['GR', 'DT', 'RHOB', 'RES'])
        ax.set_yticks(range(len(wells)))
        ax.set_yticklabels(wells)
        ax.set_title('Log Coverage Heatmap', fontweight='bold')
        plt.colorbar(im, ax=ax, label='Coverage %')

        # Add text
        for i in range(len(wells)):
            for j in range(4):
                ax.text(j, i, f'{coverages[i][j]:.0f}',
                       ha='center', va='center', fontsize=8)

        plt.tight_layout()
        fig.savefig(fig_dir / 'log_coverage.png', dpi=self.config.figure_dpi, bbox_inches='tight')
        plt.close()

    # =========================================================================
    # EXPORT
    # =========================================================================

    def export_data(self):
        """Export all data to files"""
        logger.info("Exporting data...")

        data_dir = Path(self.config.output_dir) / "data"
        data_dir.mkdir(exist_ok=True)

        # Export velocity functions
        if self.config.export_velocity_functions:
            for well_name, data in self.velocity_data.items():
                filename = data_dir / f"velocity_{well_name.replace(' ', '_')}.txt"
                with open(filename, 'w') as f:
                    f.write(f"# Velocity Function: {well_name}\n")
                    f.write("# Depth(m)\tTWT(ms)\tVelocity(m/s)\n")
                    for i in range(0, len(data['depth']), 10):  # Subsample
                        f.write(f"{data['depth'][i]:.2f}\t{data['twt'][i]:.2f}\t{data['velocity'][i]:.2f}\n")

        # Export petrophysical CSV
        if self.config.export_csv:
            for well_name, data in self.well_data.items():
                filename = data_dir / f"petrophysics_{well_name.replace(' ', '_')}.csv"
                df = pd.DataFrame({
                    'Depth_m': data['depth'],
                    'GR_API': data['gr'],
                    'DT_usm': data['dt'],
                    'RHOB_gcc': data['rhob'],
                    'Velocity_ms': data['velocity'],
                    'Vsh': data['vsh'],
                    'Phi_eff': data['phi_eff'],
                    'Sw': data['sw'] if data['sw'] is not None else np.nan,
                    'Sh': data['sh'] if data['sh'] is not None else np.nan
                })
                df.to_csv(filename, index=False, float_format='%.4f')

        # Export reservoir summary
        if self.results.reservoir_summary:
            df = pd.DataFrame(self.results.reservoir_summary)
            df.to_csv(data_dir / "reservoir_summary.csv", index=False, float_format='%.2f')

        self.results.output_files['data'] = str(data_dir)

        logger.info(f"Data exported to {data_dir}")

    # =========================================================================
    # MAIN PIPELINE
    # =========================================================================

    def run(self) -> WellResults:
        """Execute the complete well integration pipeline"""
        start_time = datetime.now()

        logger.info("=" * 80)
        logger.info("WELL INTEGRATION AUTOMATION v5.0")
        logger.info("=" * 80)

        # Initialize results
        self.results.timestamp = start_time.isoformat()
        self.results.las_directory = self.config.las_directory

        # Progress tracker
        n_steps = 6
        tracker = ProgressTracker(n_steps, self.config.webhook_url, self.config.webhook_auth)

        # Step 1: Quality assessment
        tracker.update("Quality Assessment", 15, "Analyzing LAS files")
        self.assess_quality()

        # Step 2: Process wells
        tracker.update("Well Processing", 40, "Running petrophysics")
        n_processed = self.process_wells()

        # Step 3: Velocity model
        if self.config.build_velocity_model and n_processed >= 3:
            tracker.update("Velocity Model", 60, "Building GP model")
            self.build_velocity_model()

        # Step 4: Reservoir summary
        tracker.update("Reservoir Summary", 75, "Identifying reservoirs")
        self.generate_reservoir_summary()

        # Step 5: Figures
        if self.config.save_figures:
            tracker.update("Figures", 85, "Generating visualizations")
            self.create_figures()

        # Step 6: Export
        tracker.update("Export", 95, "Exporting data")
        self.export_data()

        # Finalize
        end_time = datetime.now()
        self.results.processing_time_seconds = (end_time - start_time).total_seconds()

        # Save results
        results_file = Path(self.config.output_dir) / "well_results.json"
        self.results.to_json(str(results_file))
        self.results.output_files['results_json'] = str(results_file)

        # Save config
        config_file = Path(self.config.output_dir) / "config_used.json"
        self.config.to_json(str(config_file))

        tracker.update("Complete", 100, "Pipeline finished")

        logger.info("=" * 80)
        logger.info("PIPELINE COMPLETE")
        logger.info(f"  Wells processed: {len(self.results.wells_processed)}")
        logger.info(f"  Results: {results_file}")
        logger.info(f"  Time: {self.results.processing_time_seconds:.1f}s")
        logger.info("=" * 80)

        return self.results


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description="Well Data Integration Automation v5.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with LAS directory
    python well_integration_automation.py "path/to/las_folder" -o "outputs"

    # Run with config file
    python well_integration_automation.py -c config.json

    # Generate default config
    python well_integration_automation.py --create-config my_config.json
        """
    )

    parser.add_argument("las_directory", nargs="?", help="Directory containing LAS files")
    parser.add_argument("-o", "--output-dir", default="well_outputs", help="Output directory")
    parser.add_argument("-c", "--config", help="Configuration JSON file")
    parser.add_argument("--no-figures", action="store_true", help="Skip figure generation")
    parser.add_argument("--no-velocity-model", action="store_true", help="Skip velocity model")
    parser.add_argument("--webhook", help="Webhook URL for progress updates")
    parser.add_argument("--create-config", metavar="FILE", help="Generate default config file")

    args = parser.parse_args()

    # Generate config file
    if args.create_config:
        config = WellIntegrationConfig()
        config.to_json(args.create_config)
        print(f"Created config: {args.create_config}")
        return

    # Load or create config
    if args.config:
        config = WellIntegrationConfig.from_json(args.config)
    else:
        if not args.las_directory:
            parser.error("Either las_directory or --config is required")
        config = WellIntegrationConfig(las_directory=args.las_directory)

    # Override with CLI arguments
    if args.las_directory:
        config.las_directory = args.las_directory
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.no_figures:
        config.save_figures = False
    if args.no_velocity_model:
        config.build_velocity_model = False
    if args.webhook:
        config.webhook_url = args.webhook

    # Run
    automation = WellIntegrationAutomation(config)
    results = automation.run()

    # Print summary
    print(f"\nWells processed: {len(results.wells_processed)}")
    print(f"Reservoir intervals: {len(results.reservoir_summary)}")

    return results


if __name__ == "__main__":
    main()
