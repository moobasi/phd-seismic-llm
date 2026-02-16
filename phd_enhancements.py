"""
PhD Enhancements Module
========================

Simple additions to strengthen the PhD:
1. Monte Carlo Uncertainty for Volumetrics (P10/P50/P90)
2. 3D Visualization using Matplotlib
3. Simple Geomodeling (Property Interpolation)
4. Enhanced Fluid Contact Detection

Author: Moses Ekene Obasi
PhD Research - University of Calabar

Keep it simple - this is for PhD, not production!
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata, Rbf
from scipy import stats
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

# Base directory
BASE_DIR = Path(__file__).parent.resolve()


# =============================================================================
# 1. MONTE CARLO UNCERTAINTY FOR VOLUMETRICS
# =============================================================================

class MonteCarloVolumetrics:
    """
    Monte Carlo simulation for STOIIP/GIIP uncertainty
    Provides P10/P50/P90 estimates - essential for PhD!
    """

    def __init__(self, n_simulations: int = 10000):
        self.n_simulations = n_simulations
        self.results = None

    def stoiip_uncertainty(self,
                           area_km2: Tuple[float, float, float],      # (min, most_likely, max)
                           net_pay_m: Tuple[float, float, float],
                           porosity: Tuple[float, float, float],      # as fractions
                           sw: Tuple[float, float, float],
                           bo: Tuple[float, float, float] = (1.1, 1.2, 1.3),
                           ntg: Tuple[float, float, float] = (0.5, 0.6, 0.7)
                           ) -> Dict:
        """
        Monte Carlo STOIIP calculation with triangular distributions

        Parameters are tuples of (min, mode, max) for triangular distribution
        Returns P10, P50, P90 and full distribution
        """
        # Generate random samples using triangular distribution
        area_samples = np.random.triangular(area_km2[0], area_km2[1], area_km2[2], self.n_simulations)
        pay_samples = np.random.triangular(net_pay_m[0], net_pay_m[1], net_pay_m[2], self.n_simulations)
        por_samples = np.random.triangular(porosity[0], porosity[1], porosity[2], self.n_simulations)
        sw_samples = np.random.triangular(sw[0], sw[1], sw[2], self.n_simulations)
        bo_samples = np.random.triangular(bo[0], bo[1], bo[2], self.n_simulations)
        ntg_samples = np.random.triangular(ntg[0], ntg[1], ntg[2], self.n_simulations)

        # Calculate STOIIP for each simulation
        # STOIIP = Area * Pay * NTG * Porosity * (1-Sw) / Bo * 6.2898 (m³ to bbl)
        area_m2 = area_samples * 1e6
        grv = area_m2 * pay_samples
        nrv = grv * ntg_samples
        hcpv = nrv * por_samples * (1 - sw_samples)
        stoiip_bbl = (hcpv * 6.2898) / bo_samples
        stoiip_mmstb = stoiip_bbl / 1e6

        # Calculate percentiles
        p10 = np.percentile(stoiip_mmstb, 10)
        p50 = np.percentile(stoiip_mmstb, 50)
        p90 = np.percentile(stoiip_mmstb, 90)
        mean = np.mean(stoiip_mmstb)
        std = np.std(stoiip_mmstb)

        self.results = {
            'type': 'STOIIP',
            'n_simulations': self.n_simulations,
            'P10_mmstb': round(p10, 2),
            'P50_mmstb': round(p50, 2),
            'P90_mmstb': round(p90, 2),
            'Mean_mmstb': round(mean, 2),
            'StdDev_mmstb': round(std, 2),
            'distribution': stoiip_mmstb,
            'inputs': {
                'area_km2': area_km2,
                'net_pay_m': net_pay_m,
                'porosity': porosity,
                'sw': sw,
                'bo': bo,
                'ntg': ntg
            }
        }

        return self.results

    def giip_uncertainty(self,
                         area_km2: Tuple[float, float, float],
                         net_pay_m: Tuple[float, float, float],
                         porosity: Tuple[float, float, float],
                         sw: Tuple[float, float, float],
                         bg: Tuple[float, float, float] = (0.004, 0.005, 0.006)
                         ) -> Dict:
        """Monte Carlo GIIP calculation"""
        area_samples = np.random.triangular(area_km2[0], area_km2[1], area_km2[2], self.n_simulations)
        pay_samples = np.random.triangular(net_pay_m[0], net_pay_m[1], net_pay_m[2], self.n_simulations)
        por_samples = np.random.triangular(porosity[0], porosity[1], porosity[2], self.n_simulations)
        sw_samples = np.random.triangular(sw[0], sw[1], sw[2], self.n_simulations)
        bg_samples = np.random.triangular(bg[0], bg[1], bg[2], self.n_simulations)

        area_m2 = area_samples * 1e6
        grv = area_m2 * pay_samples
        hcpv = grv * por_samples * (1 - sw_samples)
        giip_scf = (hcpv * 35.3147) / bg_samples
        giip_bcf = giip_scf / 1e9

        p10 = np.percentile(giip_bcf, 10)
        p50 = np.percentile(giip_bcf, 50)
        p90 = np.percentile(giip_bcf, 90)

        return {
            'type': 'GIIP',
            'P10_bcf': round(p10, 2),
            'P50_bcf': round(p50, 2),
            'P90_bcf': round(p90, 2),
            'Mean_bcf': round(np.mean(giip_bcf), 2)
        }

    def plot_distribution(self, save_path: str = None):
        """Plot the STOIIP/GIIP distribution histogram"""
        if self.results is None:
            print("Run stoiip_uncertainty() or giip_uncertainty() first")
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        data = self.results['distribution']
        unit = 'MMSTB' if self.results['type'] == 'STOIIP' else 'BCF'

        ax.hist(data, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='white')

        # Add percentile lines
        p10 = self.results.get('P10_mmstb', self.results.get('P10_bcf'))
        p50 = self.results.get('P50_mmstb', self.results.get('P50_bcf'))
        p90 = self.results.get('P90_mmstb', self.results.get('P90_bcf'))

        ax.axvline(p10, color='red', linestyle='--', linewidth=2, label=f'P10: {p10:.1f} {unit}')
        ax.axvline(p50, color='green', linestyle='-', linewidth=2, label=f'P50: {p50:.1f} {unit}')
        ax.axvline(p90, color='orange', linestyle='--', linewidth=2, label=f'P90: {p90:.1f} {unit}')

        ax.set_xlabel(f'{self.results["type"]} ({unit})', fontsize=12)
        ax.set_ylabel('Probability Density', fontsize=12)
        ax.set_title(f'{self.results["type"]} Uncertainty Distribution\n(n={self.n_simulations:,} simulations)', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")

        plt.show()
        return fig


# =============================================================================
# 2. SIMPLE 3D VISUALIZATION
# =============================================================================

class Simple3DVisualization:
    """
    Simple 3D visualization using matplotlib
    Good enough for PhD - no need for complex software!
    """

    def __init__(self):
        self.fig = None
        self.ax = None

    def plot_horizon_surface(self, inline: np.ndarray, xline: np.ndarray,
                             depth: np.ndarray, title: str = "Horizon Surface",
                             save_path: str = None):
        """
        Plot a horizon surface in 3D

        Parameters:
        - inline: 1D array of inline numbers
        - xline: 1D array of xline numbers
        - depth: 2D array of depth/time values (shape: len(inline) x len(xline))
        """
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        # Create meshgrid
        IL, XL = np.meshgrid(inline, xline, indexing='ij')

        # Plot surface
        surf = ax.plot_surface(IL, XL, depth, cmap='viridis_r',
                               edgecolor='none', alpha=0.8)

        ax.set_xlabel('Inline', fontsize=11)
        ax.set_ylabel('Xline', fontsize=11)
        ax.set_zlabel('Depth/Time', fontsize=11)
        ax.set_title(title, fontsize=14)

        # Invert z-axis (depth increases downward)
        ax.invert_zaxis()

        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Depth/Time')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")

        plt.show()
        return fig

    def plot_property_cube(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                           values: np.ndarray, title: str = "Property Distribution",
                           property_name: str = "Property", save_path: str = None):
        """
        Plot 3D scatter of property values (e.g., porosity, saturation)
        """
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        scatter = ax.scatter(x, y, z, c=values, cmap='jet', alpha=0.6, s=20)

        ax.set_xlabel('X (m)', fontsize=11)
        ax.set_ylabel('Y (m)', fontsize=11)
        ax.set_zlabel('Depth (m)', fontsize=11)
        ax.set_title(title, fontsize=14)
        ax.invert_zaxis()

        fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10, label=property_name)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()
        return fig

    def plot_well_locations_3d(self, wells: Dict, horizon_depth: np.ndarray = None,
                                save_path: str = None):
        """
        Plot well locations in 3D with optional horizon surface
        """
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        # Plot wells as vertical lines
        colors = plt.cm.Set1(np.linspace(0, 1, len(wells)))

        for i, (well_name, well_data) in enumerate(wells.items()):
            if isinstance(well_data, pd.DataFrame):
                x = well_data.get('X', np.zeros(len(well_data)))
                y = well_data.get('Y', np.ones(len(well_data)) * i * 1000)
                z = well_data['Depth_m'].values

                ax.plot([i*5000]*len(z), [i*5000]*len(z), z,
                       color=colors[i], linewidth=3, label=well_name)

                # Mark top and bottom
                ax.scatter([i*5000], [i*5000], [z.min()], color=colors[i], s=100, marker='^')
                ax.scatter([i*5000], [i*5000], [z.max()], color=colors[i], s=100, marker='v')

        ax.set_xlabel('X', fontsize=11)
        ax.set_ylabel('Y', fontsize=11)
        ax.set_zlabel('Depth (m)', fontsize=11)
        ax.set_title('Well Locations', fontsize=14)
        ax.invert_zaxis()
        ax.legend(loc='upper left')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()
        return fig


# =============================================================================
# 3. SIMPLE GEOMODELING (Property Interpolation)
# =============================================================================

class SimpleGeomodel:
    """
    Simple geomodeling using well data interpolation
    Creates basic property maps - sufficient for PhD!
    """

    def __init__(self):
        self.wells = {}
        self.well_locations = {}
        self.grid = None
        self.properties = {}

    def load_wells(self, well_data_dir: str, locations_file: str = None):
        """Load well petrophysics data and locations"""
        well_dir = Path(well_data_dir)

        # Load well log data
        for csv_file in well_dir.glob("petrophysics_*.csv"):
            well_name = csv_file.stem.replace("petrophysics_", "")
            self.wells[well_name] = pd.read_csv(csv_file)

        # Load well locations from JSON file
        if locations_file is None:
            locations_file = BASE_DIR / "well_locations.json"

        if Path(locations_file).exists():
            with open(locations_file, 'r') as f:
                loc_data = json.load(f)
                self.well_locations = loc_data.get('wells', {})
            print(f"Loaded {len(self.wells)} wells with {len(self.well_locations)} locations")
        else:
            print(f"Warning: well_locations.json not found. Using pseudo-coordinates.")
            # Create pseudo-coordinates as fallback
            for i, well_name in enumerate(self.wells.keys()):
                self.well_locations[well_name] = {
                    'easting': (i % 3) * 20000 + 330000,
                    'northing': (i // 3) * 15000 + 1425000,
                    'estimated': True
                }

        return self.wells

    def get_well_xy(self, well_name: str) -> Tuple[float, float]:
        """Get X, Y coordinates for a well"""
        if well_name in self.well_locations:
            loc = self.well_locations[well_name]
            return loc.get('easting', 350000), loc.get('northing', 1440000)
        # Fallback for wells not in locations file
        i = list(self.wells.keys()).index(well_name) if well_name in self.wells else 0
        return (i % 3) * 20000 + 330000, (i // 3) * 15000 + 1425000

    def create_property_map(self, property_name: str, depth_slice: float,
                            depth_tolerance: float = 50,
                            grid_size: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create a property map at a specific depth by interpolating between wells

        Parameters:
        - property_name: 'Phi_eff', 'Sw', 'Sh', 'Vsh', etc.
        - depth_slice: Target depth in meters
        - depth_tolerance: Window around target depth
        - grid_size: Output grid resolution

        Returns:
        - X grid, Y grid, Property values
        """
        # Collect data points from all wells at target depth
        points = []
        values = []

        for i, (well_name, df) in enumerate(self.wells.items()):
            if property_name not in df.columns:
                continue

            # Filter to depth window
            mask = (df['Depth_m'] >= depth_slice - depth_tolerance) & \
                   (df['Depth_m'] <= depth_slice + depth_tolerance)

            if mask.sum() > 0:
                avg_value = df.loc[mask, property_name].mean()
                # Use actual well coordinates from locations file
                x, y = self.get_well_xy(well_name)
                points.append([x, y])
                values.append(avg_value)

        if len(points) < 3:
            print(f"Not enough wells with {property_name} data at depth {depth_slice}m")
            return None, None, None

        points = np.array(points)
        values = np.array(values)

        # Create interpolation grid
        x_min, x_max = points[:, 0].min() - 5000, points[:, 0].max() + 5000
        y_min, y_max = points[:, 1].min() - 5000, points[:, 1].max() + 5000

        xi = np.linspace(x_min, x_max, grid_size)
        yi = np.linspace(y_min, y_max, grid_size)
        XI, YI = np.meshgrid(xi, yi)

        # Interpolate using RBF (Radial Basis Function)
        try:
            rbf = Rbf(points[:, 0], points[:, 1], values, function='linear')
            ZI = rbf(XI, YI)
        except:
            # Fallback to griddata
            ZI = griddata(points, values, (XI, YI), method='linear')

        self.properties[property_name] = {
            'X': XI, 'Y': YI, 'values': ZI,
            'depth': depth_slice, 'wells_used': len(points)
        }

        return XI, YI, ZI

    def plot_property_map(self, property_name: str, depth_slice: float = None,
                          title: str = None, save_path: str = None):
        """Plot a property map"""
        if property_name not in self.properties:
            if depth_slice is None:
                print(f"Run create_property_map('{property_name}', depth) first")
                return
            self.create_property_map(property_name, depth_slice)

        data = self.properties[property_name]
        XI, YI, ZI = data['X'], data['Y'], data['values']

        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot filled contours
        contour = ax.contourf(XI, YI, ZI, levels=20, cmap='viridis')
        ax.contour(XI, YI, ZI, levels=10, colors='black', linewidths=0.5, alpha=0.5)

        # Add well locations with actual coordinates
        for well_name in self.wells.keys():
            x, y = self.get_well_xy(well_name)
            within_3d = self.well_locations.get(well_name, {}).get('within_3d', True)
            color = 'green' if within_3d else 'red'
            marker = 'o' if within_3d else '^'
            ax.plot(x, y, marker, color=color, markersize=10, markeredgecolor='black')
            ax.annotate(well_name, (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8)

        cbar = fig.colorbar(contour, ax=ax)
        cbar.set_label(property_name, fontsize=11)

        if title is None:
            title = f'{property_name} Map at {data["depth"]}m'
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Easting (m)', fontsize=11)
        ax.set_ylabel('Northing (m)', fontsize=11)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")

        plt.show()
        return fig


# =============================================================================
# 4. ENHANCED FLUID CONTACT DETECTION
# =============================================================================

class FluidContactDetector:
    """
    Enhanced fluid contact detection from well logs
    Detects OWC, GOC, GWC with confidence levels
    """

    def __init__(self):
        self.contacts = {}

    def detect_contacts(self, df: pd.DataFrame, well_name: str,
                        sw_oil_cutoff: float = 0.5,
                        sw_water_cutoff: float = 0.8,
                        porosity_cutoff: float = 0.08,
                        min_zone_thickness: int = 5) -> List[Dict]:
        """
        Detect fluid contacts from well log data

        Parameters:
        - df: DataFrame with Depth_m, Sw, Sh, Phi_eff, RHOB_gcc (optional)
        - sw_oil_cutoff: Sw below this = hydrocarbon zone
        - sw_water_cutoff: Sw above this = water zone
        - porosity_cutoff: Minimum porosity to be reservoir
        - min_zone_thickness: Minimum samples for a valid zone

        Returns:
        - List of detected contacts with depths and confidence
        """
        if 'Sw' not in df.columns or 'Depth_m' not in df.columns:
            return []

        df = df.sort_values('Depth_m').copy()
        contacts = []

        # Classify each sample
        df['Zone'] = 'transition'
        df.loc[df['Sw'] <= sw_oil_cutoff, 'Zone'] = 'hydrocarbon'
        df.loc[df['Sw'] >= sw_water_cutoff, 'Zone'] = 'water'

        # Skip non-reservoir (tight) zones
        if 'Phi_eff' in df.columns:
            df.loc[df['Phi_eff'] < porosity_cutoff, 'Zone'] = 'tight'

        # Detect if gas or oil (using density if available)
        if 'RHOB_gcc' in df.columns:
            hc_mask = df['Zone'] == 'hydrocarbon'
            df.loc[hc_mask & (df['RHOB_gcc'] < 2.0), 'Zone'] = 'gas'
            df.loc[hc_mask & (df['RHOB_gcc'] >= 2.0), 'Zone'] = 'oil'

        # Find transitions
        df['Zone_Shift'] = df['Zone'].shift(1)
        transitions = df[df['Zone'] != df['Zone_Shift']].copy()

        for idx, row in transitions.iterrows():
            prev_zone = row['Zone_Shift']
            curr_zone = row['Zone']
            depth = row['Depth_m']

            # Skip tight zones
            if 'tight' in [prev_zone, curr_zone]:
                continue
            if 'transition' in [prev_zone, curr_zone]:
                continue

            # Determine contact type
            contact_type = None
            confidence = 'MEDIUM'

            if prev_zone == 'gas' and curr_zone == 'oil':
                contact_type = 'GOC'
                confidence = 'HIGH'
            elif prev_zone == 'oil' and curr_zone == 'gas':
                contact_type = 'GOC'
                confidence = 'HIGH'
            elif prev_zone == 'oil' and curr_zone == 'water':
                contact_type = 'OWC'
                confidence = 'HIGH'
            elif prev_zone == 'water' and curr_zone == 'oil':
                contact_type = 'OWC'
                confidence = 'MEDIUM'
            elif prev_zone == 'gas' and curr_zone == 'water':
                contact_type = 'GWC'
                confidence = 'HIGH'
            elif prev_zone == 'water' and curr_zone == 'gas':
                contact_type = 'GWC'
                confidence = 'MEDIUM'
            elif prev_zone == 'hydrocarbon' and curr_zone == 'water':
                contact_type = 'HWC'  # Hydrocarbon-Water (unknown type)
                confidence = 'LOW'

            if contact_type:
                contacts.append({
                    'well': well_name,
                    'type': contact_type,
                    'depth_m': round(depth, 1),
                    'from_zone': prev_zone,
                    'to_zone': curr_zone,
                    'confidence': confidence,
                    'sw_at_contact': round(row['Sw'], 3)
                })

        # Remove duplicate contact types, keep first occurrence
        seen = set()
        unique_contacts = []
        for c in contacts:
            key = (c['type'], round(c['depth_m'], -1))  # Round to 10m
            if key not in seen:
                seen.add(key)
                unique_contacts.append(c)

        self.contacts[well_name] = unique_contacts
        return unique_contacts

    def detect_all_wells(self, well_data_dir: str) -> Dict:
        """Detect contacts in all wells"""
        well_dir = Path(well_data_dir)
        all_contacts = {}

        for csv_file in well_dir.glob("petrophysics_*.csv"):
            well_name = csv_file.stem.replace("petrophysics_", "")
            df = pd.read_csv(csv_file)
            contacts = self.detect_contacts(df, well_name)
            if contacts:
                all_contacts[well_name] = contacts

        self.contacts = all_contacts
        return all_contacts

    def get_regional_contacts(self) -> Dict:
        """Average contacts across wells for regional interpretation"""
        contact_depths = {'OWC': [], 'GOC': [], 'GWC': [], 'HWC': []}

        for well_name, contacts in self.contacts.items():
            for c in contacts:
                if c['type'] in contact_depths:
                    contact_depths[c['type']].append(c['depth_m'])

        regional = {}
        for contact_type, depths in contact_depths.items():
            if depths:
                regional[contact_type] = {
                    'mean_depth_m': round(np.mean(depths), 1),
                    'min_depth_m': round(min(depths), 1),
                    'max_depth_m': round(max(depths), 1),
                    'n_wells': len(depths)
                }

        return regional

    def generate_report(self) -> str:
        """Generate fluid contact report"""
        lines = []
        lines.append("=" * 60)
        lines.append("FLUID CONTACT ANALYSIS REPORT")
        lines.append("=" * 60)

        for well_name, contacts in self.contacts.items():
            if contacts:
                lines.append(f"\nWELL: {well_name}")
                lines.append("-" * 40)
                for c in contacts:
                    lines.append(f"  {c['type']}: {c['depth_m']}m ({c['confidence']} confidence)")
                    lines.append(f"       Transition: {c['from_zone']} -> {c['to_zone']}")

        # Regional summary
        regional = self.get_regional_contacts()
        if regional:
            lines.append("\n" + "=" * 60)
            lines.append("REGIONAL CONTACT SUMMARY")
            lines.append("=" * 60)
            for contact_type, data in regional.items():
                lines.append(f"\n{contact_type}:")
                lines.append(f"  Mean Depth: {data['mean_depth_m']}m")
                lines.append(f"  Range: {data['min_depth_m']} - {data['max_depth_m']}m")
                lines.append(f"  Observed in: {data['n_wells']} wells")

        return "\n".join(lines)


# =============================================================================
# 5. PROSPECT RANKING
# =============================================================================

class ProspectRanker:
    """
    Simple prospect ranking system
    Combines structure, reservoir, and fluid indicators
    """

    def __init__(self):
        self.prospects = []

    def add_prospect(self, name: str,
                     inline: int, xline: int,
                     closure_area_km2: float,
                     closure_relief_ms: float,
                     avg_porosity: float,
                     avg_sw: float,
                     has_dhi: bool = False,
                     near_well: bool = False,
                     notes: str = "") -> Dict:
        """Add a prospect for ranking"""
        # Calculate scores (0-100 for each factor)
        structure_score = min(100, closure_relief_ms * 1.0)  # 100ms = 100 points
        size_score = min(100, closure_area_km2 * 5)  # 20km² = 100 points
        reservoir_score = min(100, avg_porosity * 500)  # 20% = 100 points
        saturation_score = min(100, (1 - avg_sw) * 133)  # 75% HC = 100 points
        dhi_score = 100 if has_dhi else 0
        calibration_score = 50 if near_well else 0

        # Weighted total
        total_score = (
            structure_score * 0.25 +
            size_score * 0.15 +
            reservoir_score * 0.20 +
            saturation_score * 0.20 +
            dhi_score * 0.15 +
            calibration_score * 0.05
        )

        prospect = {
            'name': name,
            'inline': inline,
            'xline': xline,
            'area_km2': closure_area_km2,
            'relief_ms': closure_relief_ms,
            'porosity': avg_porosity,
            'sw': avg_sw,
            'so': 1 - avg_sw,
            'has_dhi': has_dhi,
            'near_well': near_well,
            'scores': {
                'structure': round(structure_score, 1),
                'size': round(size_score, 1),
                'reservoir': round(reservoir_score, 1),
                'saturation': round(saturation_score, 1),
                'dhi': dhi_score,
                'calibration': calibration_score
            },
            'total_score': round(total_score, 1),
            'rank': None,
            'notes': notes
        }

        self.prospects.append(prospect)
        return prospect

    def rank_prospects(self) -> List[Dict]:
        """Rank all prospects by score"""
        sorted_prospects = sorted(self.prospects, key=lambda x: x['total_score'], reverse=True)

        for i, p in enumerate(sorted_prospects):
            p['rank'] = i + 1

        self.prospects = sorted_prospects
        return sorted_prospects

    def generate_report(self) -> str:
        """Generate prospect ranking report"""
        self.rank_prospects()

        lines = []
        lines.append("=" * 70)
        lines.append("PROSPECT RANKING REPORT")
        lines.append("Bornu Chad Basin - PhD Research")
        lines.append("=" * 70)

        for p in self.prospects:
            lines.append(f"\n{'#' + str(p['rank'])} - {p['name']} (Score: {p['total_score']}/100)")
            lines.append("-" * 50)
            lines.append(f"Location: IL={p['inline']}, XL={p['xline']}")
            lines.append(f"Closure: {p['area_km2']} km², {p['relief_ms']} ms relief")
            lines.append(f"Reservoir: {p['porosity']*100:.1f}% porosity, {p['so']*100:.1f}% HC saturation")
            lines.append(f"DHI: {'Yes' if p['has_dhi'] else 'No'}")
            lines.append(f"Near Well: {'Yes' if p['near_well'] else 'No'}")
            if p['notes']:
                lines.append(f"Notes: {p['notes']}")

        return "\n".join(lines)


# =============================================================================
# MAIN - Run all analyses
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PhD ENHANCEMENTS - Running All Analyses")
    print("=" * 60)

    # 1. Monte Carlo Volumetrics
    print("\n1. MONTE CARLO UNCERTAINTY ANALYSIS")
    print("-" * 40)

    mc = MonteCarloVolumetrics(n_simulations=10000)

    # Example prospect with uncertainty ranges (min, mode, max)
    results = mc.stoiip_uncertainty(
        area_km2=(10, 15, 20),           # 10-20 km², most likely 15
        net_pay_m=(30, 50, 70),          # 30-70m, most likely 50m
        porosity=(0.15, 0.20, 0.25),     # 15-25%, most likely 20%
        sw=(0.20, 0.25, 0.35),           # 20-35%, most likely 25%
        bo=(1.1, 1.2, 1.3),
        ntg=(0.5, 0.6, 0.7)
    )

    print(f"STOIIP Results (10,000 simulations):")
    print(f"  P10 (Low):  {results['P10_mmstb']:.1f} MMSTB")
    print(f"  P50 (Mid):  {results['P50_mmstb']:.1f} MMSTB")
    print(f"  P90 (High): {results['P90_mmstb']:.1f} MMSTB")
    print(f"  Mean:       {results['Mean_mmstb']:.1f} MMSTB")

    # Save plot
    output_dir = BASE_DIR / "outputs" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    mc.plot_distribution(save_path=str(output_dir / "stoiip_uncertainty.png"))

    # 2. Fluid Contact Detection
    print("\n2. FLUID CONTACT DETECTION")
    print("-" * 40)

    detector = FluidContactDetector()
    well_data_dir = BASE_DIR / "well_outputs" / "data"

    if well_data_dir.exists():
        contacts = detector.detect_all_wells(str(well_data_dir))
        print(detector.generate_report())

        # Save report
        report_path = BASE_DIR / "outputs" / "fluid_contacts_report.txt"
        with open(report_path, 'w') as f:
            f.write(detector.generate_report())
        print(f"\nSaved: {report_path}")

    # 3. Simple Geomodel
    print("\n3. PROPERTY MAPPING (Simple Geomodel)")
    print("-" * 40)

    geomodel = SimpleGeomodel()
    locations_file = BASE_DIR / "well_locations.json"
    if well_data_dir.exists():
        geomodel.load_wells(str(well_data_dir), str(locations_file))

        # Create porosity map at 1000m depth
        XI, YI, ZI = geomodel.create_property_map('Phi_eff', depth_slice=1000, depth_tolerance=100)
        if ZI is not None:
            geomodel.plot_property_map('Phi_eff', save_path=str(output_dir / "porosity_map_1000m.png"))

        # Create Sw map
        XI, YI, ZI = geomodel.create_property_map('Sw', depth_slice=1000, depth_tolerance=100)
        if ZI is not None:
            geomodel.plot_property_map('Sw', save_path=str(output_dir / "sw_map_1000m.png"))

    # 4. Prospect Ranking Example
    print("\n4. PROSPECT RANKING")
    print("-" * 40)

    ranker = ProspectRanker()

    # Add example prospects (you would get these from horizon interpretation)
    ranker.add_prospect(
        name="Prospect Alpha",
        inline=5500, xline=5900,
        closure_area_km2=15,
        closure_relief_ms=80,
        avg_porosity=0.22,
        avg_sw=0.25,
        has_dhi=True,
        near_well=True,
        notes="Strong amplitude anomaly, near MASU-1"
    )

    ranker.add_prospect(
        name="Prospect Beta",
        inline=5800, xline=6200,
        closure_area_km2=10,
        closure_relief_ms=60,
        avg_porosity=0.18,
        avg_sw=0.30,
        has_dhi=True,
        near_well=False,
        notes="Flat spot visible"
    )

    ranker.add_prospect(
        name="Prospect Gamma",
        inline=5300, xline=5500,
        closure_area_km2=25,
        closure_relief_ms=50,
        avg_porosity=0.20,
        avg_sw=0.35,
        has_dhi=False,
        near_well=True,
        notes="Large structure, moderate relief"
    )

    print(ranker.generate_report())

    # Save report
    with open(BASE_DIR / "outputs" / "prospect_ranking.txt", 'w') as f:
        f.write(ranker.generate_report())

    print("\n" + "=" * 60)
    print("All PhD enhancements complete!")
    print("=" * 60)
