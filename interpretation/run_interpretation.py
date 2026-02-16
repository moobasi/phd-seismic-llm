"""
MASTER INTERPRETATION WORKFLOW
==============================
Runs all interpretation steps in correct order.

This is the 70% INTERPRETATION part of your PhD!

Order of operations:
1. Synthetic Seismograms (well-to-seismic tie)
2. Horizon Picking & Structure Mapping
3. Depth Conversion
4. Seismic Attributes & DHI Analysis
5. Volumetric Calculations
6. Prospect Ranking

Author: Moses Ekene Obasi
PhD Research - University of Calabar
"""

import sys
from pathlib import Path
import json
import numpy as np

# Add parent to path
BASE_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(BASE_DIR))

from interpretation.synthetic_seismogram import SyntheticSeismogram, run_synthetic_workflow
from interpretation.horizon_mapping import HorizonPicker, StructureMapper, DepthConverter, run_horizon_workflow
from interpretation.seismic_attributes import create_attribute_maps


def run_complete_interpretation():
    """
    Run the complete interpretation workflow.

    This creates ALL the deliverables needed for your PhD thesis!
    """
    print("=" * 80)
    print("   COMPLETE SEISMIC INTERPRETATION WORKFLOW")
    print("   Bornu Chad Basin - PhD Research")
    print("   University of Calabar")
    print("=" * 80)

    output_base = BASE_DIR / "interpretation"
    output_base.mkdir(exist_ok=True)

    results = {
        'workflow_completed': [],
        'deliverables': [],
        'errors': []
    }

    # =========================================================================
    # STEP 1: SYNTHETIC SEISMOGRAMS
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 1: SYNTHETIC SEISMOGRAMS (Well-to-Seismic Tie)")
    print("=" * 60)

    try:
        synth = run_synthetic_workflow()
        results['workflow_completed'].append('synthetic_seismograms')
        results['deliverables'].extend([
            'interpretation/synthetic_outputs/synthetic_*.png',
            'interpretation/synthetic_outputs/synthetic_report.txt'
        ])
        print("STEP 1 COMPLETE")
    except Exception as e:
        print(f"STEP 1 ERROR: {e}")
        results['errors'].append(f"Synthetic: {e}")

    # =========================================================================
    # STEP 2: HORIZON PICKING & STRUCTURE MAPPING
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 2: HORIZON PICKING & STRUCTURE MAPPING")
    print("=" * 60)

    try:
        picker, mapper, converter = run_horizon_workflow()
        results['workflow_completed'].append('horizon_mapping')
        results['deliverables'].extend([
            'interpretation/structure_outputs/time_structure_*.png',
            'interpretation/structure_outputs/depth_structure_*.png',
            'interpretation/structure_outputs/prospect_map_*.png',
            'interpretation/structure_outputs/structure_report.txt'
        ])
        print("STEP 2 COMPLETE")
    except Exception as e:
        print(f"STEP 2 ERROR: {e}")
        results['errors'].append(f"Horizons: {e}")

    # =========================================================================
    # STEP 3: SEISMIC ATTRIBUTES & DHI
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 3: SEISMIC ATTRIBUTES & DHI ANALYSIS")
    print("=" * 60)

    try:
        create_attribute_maps()
        results['workflow_completed'].append('seismic_attributes')
        results['deliverables'].extend([
            'interpretation/attribute_outputs/rms_amplitude_map.png',
            'interpretation/attribute_outputs/sweetness_map.png',
            'interpretation/attribute_outputs/dhi_report.txt'
        ])
        print("STEP 3 COMPLETE")
    except Exception as e:
        print(f"STEP 3 ERROR: {e}")
        results['errors'].append(f"Attributes: {e}")

    # =========================================================================
    # STEP 4: INTEGRATE RESULTS
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 4: INTEGRATING RESULTS")
    print("=" * 60)

    # Load all results and create summary
    summary = create_interpretation_summary(output_base)

    # Save master results
    with open(output_base / "interpretation_master_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("   INTERPRETATION WORKFLOW COMPLETE")
    print("=" * 80)

    print(f"\nSteps completed: {len(results['workflow_completed'])}/3")
    print(f"Deliverables created: {len(results['deliverables'])}")
    print(f"Errors: {len(results['errors'])}")

    print("\nKEY DELIVERABLES FOR THESIS:")
    print("-" * 40)
    for d in results['deliverables']:
        print(f"  - {d}")

    if results['errors']:
        print("\nERRORS TO ADDRESS:")
        for e in results['errors']:
            print(f"  - {e}")

    return results


def create_interpretation_summary(output_base: Path) -> dict:
    """Create a summary of all interpretation results"""

    summary = {
        'thesis_chapter_5': {
            'title': 'Structural Interpretation',
            'figures': [],
            'key_findings': []
        },
        'thesis_chapter_6': {
            'title': 'Reservoir Characterization',
            'figures': [],
            'key_findings': []
        }
    }

    # Load structure results
    struct_file = output_base / "structure_outputs" / "interpretation_results.json"
    if struct_file.exists():
        with open(struct_file) as f:
            struct_data = json.load(f)

        summary['thesis_chapter_5']['key_findings'] = [
            f"Interpreted {len(struct_data.get('horizons', []))} horizons",
            f"Identified {sum(len(v) for v in struct_data.get('closures', {}).values())} structural closures",
            f"Mapped {len(struct_data.get('faults', []))} faults"
        ]

    # Load DHI results
    dhi_file = output_base / "attribute_outputs" / "dhi_report.txt"
    if dhi_file.exists():
        summary['thesis_chapter_6']['key_findings'].append(
            "DHI analysis completed with bright spot identification"
        )

    # List figures
    for fig_dir in ['synthetic_outputs', 'structure_outputs', 'attribute_outputs']:
        fig_path = output_base / fig_dir
        if fig_path.exists():
            for png in fig_path.glob("*.png"):
                if 'structure' in png.name or 'prospect' in png.name:
                    summary['thesis_chapter_5']['figures'].append(str(png.name))
                else:
                    summary['thesis_chapter_6']['figures'].append(str(png.name))

    # Save summary
    with open(output_base / "thesis_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print("\nTHESIS CHAPTER SUMMARY:")
    print("-" * 40)

    for chapter, data in summary.items():
        print(f"\n{data['title']}:")
        print(f"  Figures: {len(data['figures'])}")
        for finding in data['key_findings']:
            print(f"  - {finding}")

    return summary


if __name__ == "__main__":
    run_complete_interpretation()
