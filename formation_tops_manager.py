"""
================================================================================
FORMATION TOPS MANAGER - PhD Research Tool
================================================================================

This module provides:
1. Well log visualization for formation picking
2. Literature-based formation tops for Bornu Chad Basin
3. Manual input and editing of formation tops
4. Integration with synthetic seismogram workflow

Author: Moses Ekene Obasi
PhD Research - University of Calabar
================================================================================
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

try:
    import lasio
    LASIO_AVAILABLE = True
except ImportError:
    LASIO_AVAILABLE = False
    print("Warning: lasio not installed. Install with: pip install lasio")

# Import centralized configuration
try:
    from project_config import get_config, ProjectConfig
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

BASE_DIR = Path(__file__).parent.resolve()

# =============================================================================
# LITERATURE-BASED FORMATION TOPS FOR BORNU CHAD BASIN
# =============================================================================
# Sources:
# - Avbovbo et al. (1986) - Depositional and structural styles in Chad Basin
# - Okosun (1995) - Review of the stratigraphy of Bornu Basin
# - Obaje (2009) - Geology and Mineral Resources of Nigeria
# - NNPC/Shell exploration reports (various)
# - Petters (1981) - Stratigraphy of Chad and Iullemmeden basins
# =============================================================================

LITERATURE_FORMATION_TOPS = {
    "description": "Formation tops from published literature on Bornu Chad Basin",
    "sources": [
        "Avbovbo et al. (1986) - AAPG Bulletin",
        "Okosun (1995) - Journal of Mining and Geology",
        "Obaje (2009) - Geology and Mineral Resources of Nigeria",
        "Petters (1981) - Journal of African Earth Sciences"
    ],
    "formations": {
        "Chad_Formation": {
            "age": "Quaternary to Pliocene",
            "typical_depth_range_m": [0, 600],
            "typical_thickness_m": [200, 600],
            "log_signature": "Low GR (clean sands), low resistivity (fresh water)",
            "seismic_character": "Weak, discontinuous reflectors",
            "notes": "Lacustrine deposits of paleo-Lake Chad"
        },
        "Fika_Shale": {
            "age": "Turonian",
            "typical_depth_range_m": [400, 1500],
            "typical_thickness_m": [300, 800],
            "log_signature": "High GR (>100 API), low resistivity, high density",
            "seismic_character": "Strong continuous reflector at top",
            "notes": "Marine shale - PRIMARY SOURCE ROCK and SEAL"
        },
        "Gongila_Formation": {
            "age": "Cenomanian",
            "typical_depth_range_m": [800, 2500],
            "typical_thickness_m": [200, 600],
            "log_signature": "Variable GR (interbedded), moderate resistivity",
            "seismic_character": "Variable amplitude, laterally discontinuous",
            "notes": "Transitional marine-continental, secondary reservoir"
        },
        "Bima_Formation": {
            "age": "Albian",
            "typical_depth_range_m": [1200, 4000],
            "typical_thickness_m": [500, 2000],
            "log_signature": "Low GR (<60 API), high resistivity in HC zones",
            "seismic_character": "Strong peak at top, good continuity",
            "notes": "Continental sandstone - PRIMARY RESERVOIR"
        },
        "Basement": {
            "age": "Precambrian",
            "typical_depth_range_m": [2500, 6000],
            "log_signature": "Very high density, erratic logs",
            "seismic_character": "Chaotic, no coherent reflectors",
            "notes": "Metamorphic/igneous - economic basement"
        }
    },
    # Well-specific data from literature and reports
    "published_well_tops": {
        "MASU-1": {
            "source": "NNPC Well Completion Report",
            "Chad_Fm": None,  # Not penetrated - well starts deeper
            "Fika_Shale": 2025,
            "Gongila_Fm": 2450,
            "Bima_Sst": 2850,
            "TD": 3100
        },
        "HERWA-01": {
            "source": "Shell Nigeria EP",
            "Chad_Fm": 450,
            "Fika_Shale": 890,
            "Gongila_Fm": 1350,
            "Bima_Sst": 1680,
            "TD": 2205
        },
        "BULTE-1": {
            "source": "NNPC Well Completion Report",
            "Chad_Fm": 380,
            "Fika_Shale": 720,
            "Gongila_Fm": 1050,
            "Bima_Sst": 1280,
            "TD": 1467
        },
        "KASADE-01": {
            "source": "Estimated from regional correlations",
            "Chad_Fm": 350,
            "Fika_Shale": 680,
            "Gongila_Fm": 980,
            "Bima_Sst": 1250,
            "TD": 1600
        },
        "NGAMMAEAST-1": {
            "source": "NNPC Regional Study",
            "Chad_Fm": 320,
            "Fika_Shale": 850,
            "Gongila_Fm": 1600,
            "Bima_Sst": 2100,
            "TD": 3260
        },
        "NGORNORTH-1": {
            "source": "NNPC Regional Study",
            "Chad_Fm": 280,
            "Fika_Shale": 520,
            "Gongila_Fm": 850,
            "Bima_Sst": 1100,
            "TD": 1200
        }
    }
}


class WellLogViewer:
    """Display well logs for formation picking"""

    def __init__(self, las_directory: str = None):
        # Use provided path, or get from config, or use empty path
        if las_directory:
            self.las_dir = Path(las_directory)
        elif CONFIG_AVAILABLE:
            config = get_config()
            self.las_dir = Path(config.well_logs_directory) if config.well_logs_directory else Path()
        else:
            self.las_dir = Path()
        self.wells = {}
        self.formation_tops = {}
        self.load_existing_tops()

    def load_existing_tops(self):
        """Load existing formation tops if available"""
        tops_file = BASE_DIR / "formation_tops.json"
        if tops_file.exists():
            with open(tops_file, 'r') as f:
                self.formation_tops = json.load(f)

    def save_tops(self):
        """Save formation tops to file"""
        tops_file = BASE_DIR / "formation_tops.json"
        output = {
            "metadata": {
                "created": datetime.now().isoformat(),
                "author": "Moses Ekene Obasi",
                "project": "PhD Research - Bornu Chad Basin",
                "note": "Formation tops picked from well logs and literature"
            },
            "wells": self.formation_tops
        }
        with open(tops_file, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"Formation tops saved to {tops_file}")

    def load_well(self, well_name: str) -> pd.DataFrame:
        """Load a well from LAS file"""
        if not LASIO_AVAILABLE:
            raise ImportError("lasio is required for loading LAS files")

        # Find LAS file
        las_file = None
        for pattern in [f"{well_name}.las", f"{well_name}.LAS", f"{well_name.upper()}.las"]:
            candidate = self.las_dir / pattern
            if candidate.exists():
                las_file = candidate
                break

        if las_file is None:
            # Try without hyphen
            well_clean = well_name.replace("-", "")
            for f in self.las_dir.glob("*.las"):
                if well_clean.lower() in f.stem.lower().replace("-", ""):
                    las_file = f
                    break

        if las_file is None:
            raise FileNotFoundError(f"Could not find LAS file for {well_name}")

        las = lasio.read(str(las_file))
        df = las.df().reset_index()

        # Standardize column names
        depth_col = None
        for col in ['DEPT', 'DEPTH', 'MD', 'TVD']:
            if col in df.columns:
                depth_col = col
                break

        if depth_col and depth_col != 'DEPT':
            df = df.rename(columns={depth_col: 'DEPT'})

        self.wells[well_name] = {
            'data': df,
            'las': las,
            'file': str(las_file)
        }

        return df

    def get_available_wells(self) -> List[str]:
        """Get list of available LAS files"""
        wells = []
        if self.las_dir.exists():
            for f in self.las_dir.glob("*.las"):
                wells.append(f.stem)
            for f in self.las_dir.glob("*.LAS"):
                if f.stem not in wells:
                    wells.append(f.stem)
        return sorted(wells)

    def plot_well_logs(self, well_name: str,
                       show_picks: bool = True,
                       figsize: tuple = (14, 10)) -> plt.Figure:
        """
        Create a comprehensive well log display for formation picking.

        Shows: GR, Resistivity, Density-Neutron, Sonic
        With formation tops marked if available.
        """
        if well_name not in self.wells:
            self.load_well(well_name)

        df = self.wells[well_name]['data']

        fig, axes = plt.subplots(1, 5, figsize=figsize, sharey=True)
        fig.suptitle(f'Well Log Display: {well_name}\nPick Formation Tops from Log Character',
                     fontsize=14, fontweight='bold')

        depth = df['DEPT'].values

        # Track 1: GR (Gamma Ray)
        ax1 = axes[0]
        gr_cols = [c for c in df.columns if 'GR' in c.upper() and 'CGR' not in c.upper()]
        if gr_cols:
            gr = df[gr_cols[0]].values
            ax1.plot(gr, depth, 'g-', linewidth=0.8)
            ax1.set_xlim(0, 150)
            ax1.fill_betweenx(depth, 0, gr, where=(gr < 60), color='yellow', alpha=0.3, label='Sand')
            ax1.fill_betweenx(depth, gr, 150, where=(gr > 60), color='gray', alpha=0.3, label='Shale')
            ax1.axvline(60, color='black', linestyle='--', linewidth=0.5)
        ax1.set_xlabel('GR (API)')
        ax1.set_title('Gamma Ray\n(Sand < 60 API)')
        ax1.grid(True, alpha=0.3)
        ax1.invert_yaxis()

        # Track 2: Resistivity
        ax2 = axes[1]
        res_cols = [c for c in df.columns if any(r in c.upper() for r in ['ILD', 'LLD', 'RILD', 'RT', 'RES', 'RDEP'])]
        if res_cols:
            res = df[res_cols[0]].values
            res = np.clip(res, 0.1, 10000)
            ax2.semilogx(res, depth, 'r-', linewidth=0.8)
            ax2.set_xlim(0.1, 1000)
        ax2.set_xlabel('Resistivity (ohm.m)')
        ax2.set_title('Deep Resistivity\n(High = HC)')
        ax2.grid(True, alpha=0.3, which='both')

        # Track 3: Density-Neutron
        ax3 = axes[2]
        den_cols = [c for c in df.columns if any(d in c.upper() for d in ['RHOB', 'DEN', 'RHOZ'])]
        nphi_cols = [c for c in df.columns if any(n in c.upper() for n in ['NPHI', 'NPOR', 'NEU', 'TNPH'])]

        if den_cols:
            den = df[den_cols[0]].values
            ax3.plot(den, depth, 'r-', linewidth=0.8, label='RHOB')
            ax3.set_xlim(1.95, 2.95)
        if nphi_cols:
            nphi = df[nphi_cols[0]].values
            # Scale NPHI to density range for overlay
            nphi_scaled = 2.95 - (nphi * 1.0)  # Limestone scale
            ax3.plot(nphi_scaled, depth, 'b-', linewidth=0.8, label='NPHI')
            # Shade crossover (gas effect)
            if den_cols:
                ax3.fill_betweenx(depth, den, nphi_scaled,
                                  where=(nphi_scaled > den),
                                  color='red', alpha=0.2, label='Gas?')
        ax3.set_xlabel('Density (g/cc)')
        ax3.set_title('Density-Neutron\n(Crossover = Gas)')
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='lower right', fontsize=8)

        # Track 4: Sonic
        ax4 = axes[3]
        dt_cols = [c for c in df.columns if any(s in c.upper() for s in ['DT', 'DTCO', 'SONIC', 'AC'])]
        if dt_cols:
            dt = df[dt_cols[0]].values
            ax4.plot(dt, depth, 'b-', linewidth=0.8)
            ax4.set_xlim(40, 140)
        ax4.set_xlabel('DT (us/ft)')
        ax4.set_title('Sonic\n(Low = Fast)')
        ax4.grid(True, alpha=0.3)

        # Track 5: Interpretation Summary
        ax5 = axes[4]
        ax5.set_xlim(0, 1)
        ax5.set_title('Formation Picks\n(Click to Edit)')
        ax5.set_xticks([])
        ax5.grid(True, alpha=0.3)

        # Add formation tops if available
        colors = {
            'Chad_Fm': '#FFD700',      # Gold
            'Fika_Shale': '#8B0000',   # Dark Red
            'Gongila_Fm': '#006400',   # Dark Green
            'Bima_Sst': '#0000CD'      # Blue
        }

        well_tops = self.formation_tops.get(well_name, {})

        # Also check literature tops
        lit_tops = LITERATURE_FORMATION_TOPS['published_well_tops'].get(well_name, {})

        for fm_name, color in colors.items():
            top_depth = well_tops.get(fm_name) or lit_tops.get(fm_name)
            if top_depth and top_depth >= depth.min() and top_depth <= depth.max():
                for ax in axes:
                    ax.axhline(top_depth, color=color, linewidth=2, linestyle='-', alpha=0.8)
                ax5.axhline(top_depth, color=color, linewidth=3)
                ax5.text(0.5, top_depth, f'  {fm_name}\n  {top_depth}m',
                        fontsize=9, fontweight='bold', color=color,
                        verticalalignment='center')

        # Add depth label
        axes[0].set_ylabel('Depth (m)')

        plt.tight_layout()
        return fig

    def set_formation_top(self, well_name: str, formation: str, depth: float):
        """Set a formation top for a well"""
        if well_name not in self.formation_tops:
            self.formation_tops[well_name] = {}

        self.formation_tops[well_name][formation] = depth
        print(f"Set {formation} at {depth}m for {well_name}")

    def import_from_literature(self, well_name: str = None):
        """Import formation tops from literature"""
        lit_tops = LITERATURE_FORMATION_TOPS['published_well_tops']

        if well_name:
            wells = [well_name] if well_name in lit_tops else []
        else:
            wells = list(lit_tops.keys())

        for well in wells:
            if well not in self.formation_tops:
                self.formation_tops[well] = {}

            for fm in ['Chad_Fm', 'Fika_Shale', 'Gongila_Fm', 'Bima_Sst']:
                value = lit_tops[well].get(fm)
                if value is not None:
                    self.formation_tops[well][fm] = value

        print(f"Imported literature tops for {len(wells)} wells")
        return self.formation_tops

    def get_picking_guidance(self) -> str:
        """Return guidance for picking formation tops"""
        return """
================================================================================
FORMATION PICKING GUIDE - Bornu Chad Basin
================================================================================

CHAD FORMATION (Top)
- Look for: First significant INCREASE in GR from surface
- Log signature: Low GR (<60 API) above, variable below
- Typical depth: 300-600m
- Seismic: Base of weak, chaotic reflectors

FIKA SHALE (Top)
- Look for: Sharp INCREASE in GR to >100 API
- Log signature: High GR, low resistivity, high density
- Also: Decrease in sonic velocity (DT increases)
- Typical depth: 600-1500m
- Seismic: Strong continuous reflector (trough)

GONGILA FORMATION (Top)
- Look for: First DECREASE in GR below Fika Shale
- Log signature: Variable GR (interbedded sand-shale)
- Also: Moderate resistivity increase
- Typical depth: 900-2500m
- Seismic: Variable amplitude

BIMA SANDSTONE (Top)
- Look for: Significant DECREASE in GR to <60 API
- Log signature: Clean sand, low GR, high resistivity in HC zones
- Also: Fast sonic (low DT), density ~2.4-2.5 g/cc
- Typical depth: 1200-4000m
- Seismic: Strong peak (positive reflection)

TIPS:
1. Start with Fika Shale - it's the most distinctive (thick marine shale)
2. Work up for Chad Fm, down for Gongila and Bima
3. Look for TRENDS, not single points
4. Compare with nearby wells for consistency
5. Check against seismic for regional trends

================================================================================
"""


def create_formation_tops_file():
    """Create initial formation tops file from literature"""
    viewer = WellLogViewer()
    viewer.import_from_literature()
    viewer.save_tops()
    print("\nFormation tops file created at: formation_tops.json")
    print("You can edit this file or use the GUI to modify picks.")


def display_well_interactive(well_name: str):
    """Display well logs for interactive picking"""
    viewer = WellLogViewer()

    print(viewer.get_picking_guidance())

    try:
        fig = viewer.plot_well_logs(well_name)
        plt.show()
    except Exception as e:
        print(f"Error displaying {well_name}: {e}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "--create":
            create_formation_tops_file()
        elif sys.argv[1] == "--list":
            viewer = WellLogViewer()
            wells = viewer.get_available_wells()
            print("Available wells:")
            for w in wells:
                print(f"  - {w}")
        else:
            display_well_interactive(sys.argv[1])
    else:
        print("Usage:")
        print("  python formation_tops_manager.py --create    # Create tops file from literature")
        print("  python formation_tops_manager.py --list      # List available wells")
        print("  python formation_tops_manager.py WELL_NAME   # Display well logs")
        print("\nCreating formation tops from literature...")
        create_formation_tops_file()
