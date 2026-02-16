"""
Fluid Analysis Module for Seismic Interpretation
=================================================

Discriminates Oil, Gas, and Water from well logs and seismic attributes.

Methods:
1. Resistivity-Porosity crossplot (Pickett plot)
2. Neutron-Density crossplot (gas detection)
3. Acoustic Impedance vs Porosity (fluid typing)
4. Vp/Vs ratio analysis (AVO)

Author: Moses Ekene Obasi
PhD Research - University of Calabar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

# Try GPU acceleration
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

# Import centralized configuration
try:
    from project_config import get_config, ProjectConfig
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False


class FluidAnalyzer:
    """Analyze fluid content from well logs and seismic"""

    # Fluid properties (typical values)
    FLUID_PROPERTIES = {
        'oil': {
            'density_gcc': (0.7, 0.9),
            'velocity_ms': (1200, 1500),
            'resistivity_ohm': (10, 1000),
            'ai_range': (2000, 5000)
        },
        'gas': {
            'density_gcc': (0.1, 0.3),
            'velocity_ms': (400, 800),
            'resistivity_ohm': (100, 10000),
            'ai_range': (1000, 3000)
        },
        'water': {
            'density_gcc': (1.0, 1.1),
            'velocity_ms': (1500, 1700),
            'resistivity_ohm': (0.1, 10),
            'ai_range': (3000, 7000)
        }
    }

    # Cutoffs for pay zone identification
    PAY_CUTOFFS = {
        'porosity_min': 0.10,      # 10% minimum porosity
        'sw_max': 0.50,            # 50% maximum water saturation
        'vshale_max': 0.40,        # 40% maximum shale volume
        'permeability_min': 1.0    # 1 mD minimum permeability
    }

    def __init__(self, well_data_dir: str = None):
        self.well_data_dir = Path(well_data_dir) if well_data_dir else None
        self.wells = {}
        self.results = {}

    def load_well_data(self, csv_path: str) -> pd.DataFrame:
        """Load petrophysical data from CSV"""
        df = pd.read_csv(csv_path)
        return df

    def load_all_wells(self) -> Dict[str, pd.DataFrame]:
        """Load all well data from directory"""
        if not self.well_data_dir:
            return {}

        for csv_file in self.well_data_dir.glob("petrophysics_*.csv"):
            well_name = csv_file.stem.replace("petrophysics_", "")
            self.wells[well_name] = pd.read_csv(csv_file)

        return self.wells

    def calculate_fluid_type(self, sw: float, porosity: float,
                            resistivity: float, density: float = None) -> str:
        """
        Determine fluid type from log properties

        Returns: 'oil', 'gas', 'water', or 'tight'
        """
        # Hydrocarbon saturation
        sh = 1 - sw

        # Water zone
        if sw > 0.8:
            return 'water'

        # Tight/non-reservoir
        if porosity < 0.05:
            return 'tight'

        # Gas indicators
        if density is not None and density < 2.0:
            # Low density suggests gas
            if sh > 0.3 and resistivity > 50:
                return 'gas'

        # Oil indicators
        if sh > 0.5 and resistivity > 10:
            if density is None or density >= 2.0:
                return 'oil'

        # Mixed or uncertain
        if sh > 0.2:
            return 'oil'  # Default to oil if hydrocarbons present

        return 'water'

    def analyze_well(self, df: pd.DataFrame, well_name: str = "Unknown") -> Dict:
        """
        Comprehensive fluid analysis for a well
        """
        results = {
            'well_name': well_name,
            'depth_range': (float(df['Depth_m'].min()), float(df['Depth_m'].max())),
            'total_samples': len(df),
            'zones': [],
            'summary': {}
        }

        # Required columns
        required = ['Depth_m', 'Sw', 'Sh', 'Phi_eff']
        if not all(col in df.columns for col in required):
            results['error'] = "Missing required columns"
            return results

        # Get optional columns
        has_density = 'RHOB_gcc' in df.columns
        has_resistivity = 'Resistivity' in df.columns or any('RES' in c.upper() for c in df.columns)

        # Classify each sample
        fluid_types = []
        for idx, row in df.iterrows():
            sw = row['Sw']
            porosity = row['Phi_eff']
            density = row.get('RHOB_gcc', None)
            resistivity = 100  # Default if not available

            fluid = self.calculate_fluid_type(sw, porosity, resistivity, density)
            fluid_types.append(fluid)

        df = df.copy()
        df['Fluid_Type'] = fluid_types

        # Identify zones
        zones = self._identify_zones(df)
        results['zones'] = zones

        # Summary statistics
        results['summary'] = {
            'oil_samples': int((df['Fluid_Type'] == 'oil').sum()),
            'gas_samples': int((df['Fluid_Type'] == 'gas').sum()),
            'water_samples': int((df['Fluid_Type'] == 'water').sum()),
            'tight_samples': int((df['Fluid_Type'] == 'tight').sum()),
            'avg_porosity_pct': float(df['Phi_eff'].mean() * 100),
            'avg_sw_pct': float(df['Sw'].mean() * 100),
            'avg_sh_pct': float(df['Sh'].mean() * 100),
            'max_sh_pct': float(df['Sh'].max() * 100)
        }

        # Pay zone summary
        pay_mask = (df['Sh'] > 0.5) & (df['Phi_eff'] > 0.10)
        if pay_mask.sum() > 0:
            pay_df = df[pay_mask]
            results['pay_summary'] = {
                'net_pay_samples': int(pay_mask.sum()),
                'depth_top_m': float(pay_df['Depth_m'].min()),
                'depth_base_m': float(pay_df['Depth_m'].max()),
                'avg_porosity_pct': float(pay_df['Phi_eff'].mean() * 100),
                'avg_sh_pct': float(pay_df['Sh'].mean() * 100),
                'dominant_fluid': pay_df['Fluid_Type'].mode().iloc[0] if len(pay_df) > 0 else 'unknown'
            }
        else:
            results['pay_summary'] = None

        return results

    def _identify_zones(self, df: pd.DataFrame) -> List[Dict]:
        """Identify continuous fluid zones"""
        zones = []
        current_zone = None

        for idx, row in df.iterrows():
            fluid = row['Fluid_Type']
            depth = row['Depth_m']

            if current_zone is None:
                current_zone = {
                    'fluid': fluid,
                    'top': depth,
                    'base': depth,
                    'samples': 1,
                    'avg_porosity': row['Phi_eff'],
                    'avg_sh': row['Sh']
                }
            elif fluid == current_zone['fluid']:
                current_zone['base'] = depth
                current_zone['samples'] += 1
                current_zone['avg_porosity'] += row['Phi_eff']
                current_zone['avg_sh'] += row['Sh']
            else:
                # Finalize current zone
                if current_zone['samples'] > 10:  # Minimum zone thickness
                    current_zone['avg_porosity'] /= current_zone['samples']
                    current_zone['avg_sh'] /= current_zone['samples']
                    current_zone['thickness_m'] = current_zone['base'] - current_zone['top']
                    zones.append(current_zone)

                # Start new zone
                current_zone = {
                    'fluid': fluid,
                    'top': depth,
                    'base': depth,
                    'samples': 1,
                    'avg_porosity': row['Phi_eff'],
                    'avg_sh': row['Sh']
                }

        # Add last zone
        if current_zone and current_zone['samples'] > 10:
            current_zone['avg_porosity'] /= current_zone['samples']
            current_zone['avg_sh'] /= current_zone['samples']
            current_zone['thickness_m'] = current_zone['base'] - current_zone['top']
            zones.append(current_zone)

        return zones

    def calculate_stoiip(self, area_km2: float, net_pay_m: float,
                         porosity: float = 0.20, sw: float = 0.25,
                         bo: float = 1.2, recovery_factor: float = 0.35) -> Dict:
        """
        Calculate Stock Tank Oil Initially In Place (STOIIP)

        Parameters:
        - area_km2: Closure area in km²
        - net_pay_m: Net pay thickness in meters
        - porosity: Average porosity (fraction)
        - sw: Water saturation (fraction)
        - bo: Oil formation volume factor
        - recovery_factor: Expected recovery factor

        Returns:
        - STOIIP and EUR in various units
        """
        # Convert area to m²
        area_m2 = area_km2 * 1e6

        # Gross Rock Volume
        grv_m3 = area_m2 * net_pay_m

        # Hydrocarbon Pore Volume
        hcpv_m3 = grv_m3 * porosity * (1 - sw)

        # STOIIP in barrels (1 m³ = 6.2898 bbl)
        stoiip_bbl = (hcpv_m3 * 6.2898) / bo

        # EUR
        eur_bbl = stoiip_bbl * recovery_factor

        return {
            'inputs': {
                'area_km2': area_km2,
                'net_pay_m': net_pay_m,
                'porosity': porosity,
                'sw': sw,
                'so': 1 - sw,
                'bo': bo,
                'recovery_factor': recovery_factor
            },
            'grv_m3': grv_m3,
            'grv_acre_ft': grv_m3 * 0.000810714,
            'hcpv_m3': hcpv_m3,
            'stoiip_bbl': stoiip_bbl,
            'stoiip_mmstb': stoiip_bbl / 1e6,
            'eur_bbl': eur_bbl,
            'eur_mmstb': eur_bbl / 1e6
        }

    def calculate_giip(self, area_km2: float, net_pay_m: float,
                       porosity: float = 0.20, sw: float = 0.20,
                       bg: float = 0.005, recovery_factor: float = 0.75) -> Dict:
        """
        Calculate Gas Initially In Place (GIIP)

        Parameters:
        - area_km2: Closure area in km²
        - net_pay_m: Net pay thickness in meters
        - porosity: Average porosity (fraction)
        - sw: Water saturation (fraction)
        - bg: Gas formation volume factor (res vol / std vol)
        - recovery_factor: Expected recovery factor

        Returns:
        - GIIP in BCF and EUR
        """
        area_m2 = area_km2 * 1e6
        grv_m3 = area_m2 * net_pay_m
        hcpv_m3 = grv_m3 * porosity * (1 - sw)

        # GIIP in standard cubic feet (1 m³ = 35.3147 scf at standard conditions)
        giip_scf = (hcpv_m3 * 35.3147) / bg
        giip_bcf = giip_scf / 1e9

        eur_bcf = giip_bcf * recovery_factor

        return {
            'inputs': {
                'area_km2': area_km2,
                'net_pay_m': net_pay_m,
                'porosity': porosity,
                'sw': sw,
                'sg': 1 - sw,
                'bg': bg,
                'recovery_factor': recovery_factor
            },
            'grv_m3': grv_m3,
            'hcpv_m3': hcpv_m3,
            'giip_scf': giip_scf,
            'giip_bcf': giip_bcf,
            'giip_tcf': giip_bcf / 1000,
            'eur_bcf': eur_bcf,
            'eur_tcf': eur_bcf / 1000
        }

    def generate_report(self, output_path: str = None) -> str:
        """Generate a comprehensive fluid analysis report"""
        report = []
        report.append("=" * 70)
        report.append("FLUID ANALYSIS REPORT")
        report.append("Bornu Chad Basin - PhD Research")
        report.append("=" * 70)
        report.append("")

        # Load all wells if not already loaded
        if not self.wells:
            self.load_all_wells()

        # Analyze each well
        for well_name, df in self.wells.items():
            results = self.analyze_well(df, well_name)
            self.results[well_name] = results

            report.append(f"\nWELL: {well_name}")
            report.append("-" * 50)
            report.append(f"Depth Range: {results['depth_range'][0]:.0f} - {results['depth_range'][1]:.0f} m")
            report.append(f"Total Samples: {results['total_samples']}")

            if 'summary' in results:
                s = results['summary']
                report.append(f"\nFluid Distribution:")
                report.append(f"  Oil zones:   {s['oil_samples']} samples")
                report.append(f"  Gas zones:   {s['gas_samples']} samples")
                report.append(f"  Water zones: {s['water_samples']} samples")
                report.append(f"  Tight zones: {s['tight_samples']} samples")
                report.append(f"\nReservoir Properties:")
                report.append(f"  Avg Porosity: {s['avg_porosity_pct']:.1f}%")
                report.append(f"  Avg Water Sat: {s['avg_sw_pct']:.1f}%")
                report.append(f"  Max HC Sat: {s['max_sh_pct']:.1f}%")

            if results.get('pay_summary'):
                p = results['pay_summary']
                report.append(f"\nPAY ZONE IDENTIFIED:")
                report.append(f"  Depth: {p['depth_top_m']:.0f} - {p['depth_base_m']:.0f} m")
                report.append(f"  Net Pay: {p['net_pay_samples']} samples")
                report.append(f"  Avg Porosity: {p['avg_porosity_pct']:.1f}%")
                report.append(f"  Avg HC Sat: {p['avg_sh_pct']:.1f}%")
                report.append(f"  Dominant Fluid: {p['dominant_fluid'].upper()}")

        # Overall summary
        report.append("\n" + "=" * 70)
        report.append("BASIN SUMMARY")
        report.append("=" * 70)

        total_oil = sum(r.get('summary', {}).get('oil_samples', 0) for r in self.results.values())
        total_gas = sum(r.get('summary', {}).get('gas_samples', 0) for r in self.results.values())
        total_water = sum(r.get('summary', {}).get('water_samples', 0) for r in self.results.values())

        report.append(f"Total Oil Zones: {total_oil} samples across all wells")
        report.append(f"Total Gas Zones: {total_gas} samples across all wells")
        report.append(f"Total Water Zones: {total_water} samples across all wells")

        if total_oil > total_gas:
            report.append("\nDOMINANT HYDROCARBON TYPE: OIL")
        elif total_gas > total_oil:
            report.append("\nDOMINANT HYDROCARBON TYPE: GAS")
        else:
            report.append("\nDOMINANT HYDROCARBON TYPE: MIXED OIL/GAS")

        report_text = "\n".join(report)

        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            print(f"Report saved to: {output_path}")

        return report_text


# =============================================================================
# Main execution
# =============================================================================

if __name__ == "__main__":
    from pathlib import Path

    # Get well data directory from config or use default relative path
    if CONFIG_AVAILABLE:
        config = get_config()
        well_dir = Path(config.output_directory) / "well_outputs" / "data" if config.output_directory else Path("well_outputs/data")
    else:
        well_dir = Path(__file__).parent / "well_outputs" / "data"

    # Initialize analyzer
    analyzer = FluidAnalyzer(well_data_dir=str(well_dir))

    # Generate report
    report = analyzer.generate_report()
    print(report)

    # Example STOIIP calculation
    print("\n" + "=" * 70)
    print("EXAMPLE VOLUMETRIC CALCULATION")
    print("=" * 70)

    # For a 15 km² prospect with 50m net pay
    stoiip = analyzer.calculate_stoiip(
        area_km2=15,
        net_pay_m=50,
        porosity=0.20,
        sw=0.25,
        bo=1.2,
        recovery_factor=0.35
    )

    print(f"\nProspect: 15 km² area, 50m net pay")
    print(f"Porosity: 20%, Sw: 25%, So: 75%")
    print(f"\nResults:")
    print(f"  GRV: {stoiip['grv_m3']:.2e} m³")
    print(f"  HCPV: {stoiip['hcpv_m3']:.2e} m³")
    print(f"  STOIIP: {stoiip['stoiip_mmstb']:.1f} MMSTB")
    print(f"  EUR (35%): {stoiip['eur_mmstb']:.1f} MMSTB")
