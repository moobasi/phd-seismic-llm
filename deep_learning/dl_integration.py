"""
================================================================================
DEEP LEARNING INTEGRATION MODULE
Master orchestrator for DL-powered seismic interpretation
================================================================================

Integrates all deep learning modules with the LLM assistant for
intelligent, explainable seismic interpretation.

Workflow:
1. Run DL models (fault detection, facies classification)
2. Aggregate results
3. Feed to LLM for geological interpretation
4. Generate comprehensive reports

Author: Moses Ekene Obasi
PhD Research - University of Calabar
================================================================================
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import DL modules
from .dl_fault_detection import FaultDetector, FaultDetectionConfig, FaultDetectionResults
from .dl_facies_classification import FaciesClassifier, FaciesClassificationConfig, FaciesClassificationResults

try:
    import segyio
    SEGYIO_AVAILABLE = True
except ImportError:
    SEGYIO_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class DeepLearningConfig:
    """Master configuration for deep learning interpretation"""

    # Input
    seismic_file: str = ""
    output_dir: str = "dl_interpretation"

    # Module enables
    run_fault_detection: bool = True
    run_facies_classification: bool = True

    # GPU settings
    use_gpu: bool = True

    # Sub-module configs
    fault_config: Optional[FaultDetectionConfig] = None
    facies_config: Optional[FaciesClassificationConfig] = None

    # LLM integration
    enable_llm_interpretation: bool = True
    llm_model: str = "ollama"  # ollama, anthropic, openai

    # Report generation
    generate_report: bool = True
    report_format: str = "markdown"  # markdown, html, pdf

    # Bornu Chad Basin specific
    basin_name: str = "Bornu Chad Basin"
    regional_context: str = """
    The Bornu Chad Basin is an intracratonic rift basin in northeastern Nigeria,
    part of the West and Central African Rift System (WCARS). Key formations include:
    - Chad Formation: Quaternary lacustrine/fluvial deposits
    - Fika Shale: Turonian marine shale (potential seal)
    - Gongila Formation: Cenomanian transitional marine (potential reservoir)
    - Bima Formation: Albian continental clastics (proven reservoir)
    - Basement: Pre-Cambrian crystalline rocks

    The basin exhibits NE-trending horst and graben structures from Cretaceous
    extension, with fault-controlled sedimentation patterns.
    """

    def to_dict(self) -> Dict:
        result = asdict(self)
        if self.fault_config:
            result['fault_config'] = self.fault_config.to_dict()
        if self.facies_config:
            result['facies_config'] = self.facies_config.to_dict()
        return result


@dataclass
class IntegratedResults:
    """Combined results from all DL modules"""

    success: bool = False

    # Individual results
    fault_results: Optional[FaultDetectionResults] = None
    facies_results: Optional[FaciesClassificationResults] = None

    # Integrated analysis
    fault_facies_correlation: Dict = field(default_factory=dict)
    reservoir_risk_assessment: Dict = field(default_factory=dict)
    prospect_summary: List[Dict] = field(default_factory=list)

    # LLM interpretation
    llm_geological_interpretation: str = ""
    llm_recommendations: str = ""

    # Output files
    report_file: str = ""
    combined_figure: str = ""

    # Metadata
    total_processing_time: float = 0.0
    timestamp: str = ""

    def to_dict(self) -> Dict:
        result = {
            'success': self.success,
            'fault_facies_correlation': self.fault_facies_correlation,
            'reservoir_risk_assessment': self.reservoir_risk_assessment,
            'prospect_summary': self.prospect_summary,
            'llm_geological_interpretation': self.llm_geological_interpretation,
            'llm_recommendations': self.llm_recommendations,
            'report_file': self.report_file,
            'combined_figure': self.combined_figure,
            'total_processing_time': self.total_processing_time,
            'timestamp': self.timestamp
        }
        if self.fault_results:
            result['fault_results'] = self.fault_results.to_dict()
        if self.facies_results:
            result['facies_results'] = self.facies_results.to_dict()
        return result


# =============================================================================
# MAIN INTEGRATION CLASS
# =============================================================================

class DeepLearningIntegration:
    """
    Master orchestrator for deep learning seismic interpretation.

    Runs all DL modules, integrates results, and provides LLM-powered
    geological interpretation with recommendations.

    Example:
        config = DeepLearningConfig(seismic_file="data.segy")
        integration = DeepLearningIntegration(config)
        results = integration.run_full_interpretation()

        # Get LLM interpretation
        print(results.llm_geological_interpretation)
    """

    def __init__(self, config: Optional[DeepLearningConfig] = None):
        self.config = config or DeepLearningConfig()
        self.seismic_data = None
        self.fault_detector = None
        self.facies_classifier = None

        self._setup_modules()

    def _setup_modules(self):
        """Initialize DL modules"""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Fault detection
        if self.config.run_fault_detection:
            fault_config = self.config.fault_config or FaultDetectionConfig()
            fault_config.seismic_file = self.config.seismic_file
            fault_config.output_dir = str(output_dir / "faults")
            fault_config.use_gpu = self.config.use_gpu
            self.fault_detector = FaultDetector(fault_config)

        # Facies classification
        if self.config.run_facies_classification:
            facies_config = self.config.facies_config or FaciesClassificationConfig()
            facies_config.seismic_file = self.config.seismic_file
            facies_config.output_dir = str(output_dir / "facies")
            facies_config.use_gpu = self.config.use_gpu
            self.facies_classifier = FaciesClassifier(facies_config)

    def load_seismic(self) -> bool:
        """Load seismic data"""
        if not SEGYIO_AVAILABLE:
            print("Error: segyio not available")
            return False

        try:
            with segyio.open(self.config.seismic_file, 'r', ignore_geometry=True) as f:
                self.seismic_data = segyio.tools.cube(f)
                print(f"Loaded seismic: {self.seismic_data.shape}")
                return True
        except Exception as e:
            print(f"Error loading seismic: {e}")
            return False

    def run_full_interpretation(self) -> IntegratedResults:
        """
        Run complete deep learning interpretation workflow.

        1. Load seismic data
        2. Run fault detection
        3. Run facies classification
        4. Integrate and analyze results
        5. Generate LLM interpretation
        6. Create comprehensive report

        Returns:
            IntegratedResults with all analysis and interpretations
        """
        start_time = datetime.now()
        results = IntegratedResults()
        results.timestamp = start_time.isoformat()

        print("\n" + "="*70)
        print("DEEP LEARNING SEISMIC INTERPRETATION")
        print(f"Basin: {self.config.basin_name}")
        print("="*70 + "\n")

        # Load seismic data
        print("Step 1: Loading seismic data...")
        if not self.load_seismic():
            results.llm_geological_interpretation = "Error: Could not load seismic data"
            return results

        # Run fault detection
        if self.config.run_fault_detection and self.fault_detector:
            print("\nStep 2: Running fault detection (FaultSeg3D-style)...")
            results.fault_results = self.fault_detector.detect_faults(self.seismic_data)

        # Run facies classification
        if self.config.run_facies_classification and self.facies_classifier:
            print("\nStep 3: Running facies classification (DeepSeismic-style)...")
            results.facies_results = self.facies_classifier.classify_facies(self.seismic_data)

        # Integrate results
        print("\nStep 4: Integrating results...")
        results = self._integrate_results(results)

        # Generate LLM interpretation
        if self.config.enable_llm_interpretation:
            print("\nStep 5: Generating LLM interpretation...")
            results = self._generate_llm_interpretation(results)

        # Create combined visualization
        print("\nStep 6: Creating visualizations...")
        results.combined_figure = self._create_combined_figure(results)

        # Generate report
        if self.config.generate_report:
            print("\nStep 7: Generating report...")
            results.report_file = self._generate_report(results)

        # Finalize
        results.success = True
        results.total_processing_time = (datetime.now() - start_time).total_seconds()

        # Save results
        output_dir = Path(self.config.output_dir)
        results_file = output_dir / "integrated_results.json"
        with open(results_file, 'w') as f:
            json.dump(results.to_dict(), f, indent=2, default=str)

        print("\n" + "="*70)
        print(f"Interpretation complete in {results.total_processing_time:.1f} seconds")
        print(f"Results saved to: {output_dir}")
        print("="*70 + "\n")

        return results

    def _integrate_results(self, results: IntegratedResults) -> IntegratedResults:
        """Integrate fault and facies results"""

        # Fault-facies correlation
        if results.fault_results and results.facies_results:
            # Load the numpy arrays
            fault_dir = Path(self.config.output_dir) / "faults"
            facies_dir = Path(self.config.output_dir) / "facies"

            try:
                fault_binary = np.load(fault_dir / "fault_binary.npy")
                facies_class = np.load(facies_dir / "facies_classification.npy")

                # Analyze which facies are cut by faults
                correlation = {}
                for i, class_name in enumerate(self.facies_classifier.config.class_names):
                    facies_mask = facies_class == i
                    faults_in_facies = np.sum(fault_binary[facies_mask])
                    total_facies = np.sum(facies_mask)

                    if total_facies > 0:
                        fault_percentage = 100.0 * faults_in_facies / total_facies
                        correlation[class_name] = {
                            'fault_percentage': float(fault_percentage),
                            'total_voxels': int(total_facies),
                            'faulted_voxels': int(faults_in_facies)
                        }

                results.fault_facies_correlation = correlation
            except Exception as e:
                print(f"Warning: Could not correlate fault-facies: {e}")

        # Reservoir risk assessment
        results.reservoir_risk_assessment = self._assess_reservoir_risk(results)

        # Identify prospects
        results.prospect_summary = self._identify_prospects(results)

        return results

    def _assess_reservoir_risk(self, results: IntegratedResults) -> Dict:
        """Assess reservoir risk based on DL results"""
        risk = {
            'compartmentalization_risk': 'Unknown',
            'seal_integrity': 'Unknown',
            'reservoir_quality': 'Unknown',
            'overall_risk': 'Unknown'
        }

        # Compartmentalization from fault density
        if results.fault_results:
            fault_vol = results.fault_results.fault_volume_percent
            if fault_vol > 5:
                risk['compartmentalization_risk'] = 'High'
            elif fault_vol > 2:
                risk['compartmentalization_risk'] = 'Medium'
            else:
                risk['compartmentalization_risk'] = 'Low'

        # Seal integrity from facies
        if results.facies_results:
            shale_pct = results.facies_results.class_percentages.get('Marine Shale', 0)
            if shale_pct > 30:
                risk['seal_integrity'] = 'Good'
            elif shale_pct > 15:
                risk['seal_integrity'] = 'Moderate'
            else:
                risk['seal_integrity'] = 'Poor'

            # Reservoir quality from sand content
            sand_pct = results.facies_results.class_percentages.get('Channel Sand', 0)
            if sand_pct > 20:
                risk['reservoir_quality'] = 'Good'
            elif sand_pct > 10:
                risk['reservoir_quality'] = 'Moderate'
            else:
                risk['reservoir_quality'] = 'Poor'

        # Overall risk
        risks = [risk['compartmentalization_risk'],
                 risk['seal_integrity'],
                 risk['reservoir_quality']]

        if 'Poor' in risks or 'High' in risks:
            risk['overall_risk'] = 'High'
        elif 'Moderate' in risks or 'Medium' in risks:
            risk['overall_risk'] = 'Medium'
        else:
            risk['overall_risk'] = 'Low'

        return risk

    def _identify_prospects(self, results: IntegratedResults) -> List[Dict]:
        """Identify potential prospects from integrated analysis"""
        prospects = []

        if not results.fault_results or not results.facies_results:
            return prospects

        # Simple prospect identification based on:
        # - Channel sand presence
        # - Structural closure (from fault analysis)
        # - Seal presence

        # This is a simplified version - real implementation would
        # use actual horizon maps and closure analysis

        if results.facies_results.class_percentages.get('Channel Sand', 0) > 10:
            prospects.append({
                'prospect_id': 'P1',
                'type': 'Stratigraphic',
                'reservoir': 'Channel Sand',
                'seal': 'Marine Shale',
                'risk': results.reservoir_risk_assessment.get('overall_risk', 'Unknown'),
                'notes': 'Requires closure confirmation from horizon mapping'
            })

        if results.fault_results.num_faults_detected > 0:
            prospects.append({
                'prospect_id': 'P2',
                'type': 'Structural (Fault-bounded)',
                'reservoir': 'Fault block closure',
                'seal': 'Fault seal + shale',
                'risk': results.reservoir_risk_assessment.get('compartmentalization_risk', 'Unknown'),
                'notes': 'Requires fault seal analysis'
            })

        return prospects

    def _generate_llm_interpretation(self, results: IntegratedResults) -> IntegratedResults:
        """Generate geological interpretation using LLM"""

        # Build context for LLM
        context = f"""
DEEP LEARNING SEISMIC INTERPRETATION RESULTS
=============================================

REGIONAL CONTEXT:
{self.config.regional_context}

"""
        if results.fault_results:
            context += f"""
FAULT DETECTION RESULTS:
{results.fault_results.summary_for_llm}
"""

        if results.facies_results:
            context += f"""
FACIES CLASSIFICATION RESULTS:
{results.facies_results.summary_for_llm}
"""

        context += f"""
INTEGRATED ANALYSIS:

Fault-Facies Correlation:
{json.dumps(results.fault_facies_correlation, indent=2)}

Reservoir Risk Assessment:
{json.dumps(results.reservoir_risk_assessment, indent=2)}

Identified Prospects:
{json.dumps(results.prospect_summary, indent=2)}
"""

        # Generate interpretation
        interpretation_prompt = """
Based on the deep learning interpretation results above, provide:

1. GEOLOGICAL INTERPRETATION:
   - Structural framework interpretation
   - Depositional environment analysis
   - Stratigraphic relationships

2. HYDROCARBON PROSPECTIVITY:
   - Reservoir potential
   - Seal effectiveness
   - Trap types present

3. RECOMMENDATIONS:
   - Additional analyses needed
   - Data quality concerns
   - Prospect ranking priorities

Please provide specific, technical interpretations suitable for a PhD thesis
on the Bornu Chad Basin, Nigeria.
"""

        # Try to get LLM interpretation
        llm_response = self._call_llm(context + interpretation_prompt)

        if llm_response:
            results.llm_geological_interpretation = llm_response
        else:
            # Fallback: generate template interpretation
            results.llm_geological_interpretation = self._generate_fallback_interpretation(results)

        # Generate recommendations
        results.llm_recommendations = self._generate_recommendations(results)

        return results

    def _call_llm(self, prompt: str) -> Optional[str]:
        """Call LLM for interpretation"""
        try:
            import subprocess

            # Try Ollama first
            result = subprocess.run(
                ["ollama", "run", "llama3.2", prompt],
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode == 0:
                return result.stdout.strip()
        except Exception as e:
            print(f"LLM call failed: {e}")

        return None

    def _generate_fallback_interpretation(self, results: IntegratedResults) -> str:
        """Generate interpretation without LLM"""

        interpretation = f"""
GEOLOGICAL INTERPRETATION (Automated Analysis)
==============================================

STRUCTURAL FRAMEWORK:
"""
        if results.fault_results:
            interpretation += f"""
- {results.fault_results.num_faults_detected} fault bodies detected
- Fault volume: {results.fault_results.fault_volume_percent:.2f}%
- Primary fault orientations suggest NE-SW trending structures consistent
  with the West and Central African Rift System (WCARS) extension

"""

        interpretation += """
DEPOSITIONAL ENVIRONMENT:
"""
        if results.facies_results:
            for name, pct in sorted(results.facies_results.class_percentages.items(),
                                    key=lambda x: x[1], reverse=True)[:3]:
                interpretation += f"- {name}: {pct:.1f}%\n"

            interpretation += """
The facies distribution suggests a mixed depositional system with
fluvial/lacustrine and marine influences, consistent with the
Bornu Chad Basin's evolution from continental rift to marine incursion.

"""

        interpretation += f"""
RESERVOIR RISK ASSESSMENT:
- Compartmentalization Risk: {results.reservoir_risk_assessment.get('compartmentalization_risk', 'Unknown')}
- Seal Integrity: {results.reservoir_risk_assessment.get('seal_integrity', 'Unknown')}
- Reservoir Quality: {results.reservoir_risk_assessment.get('reservoir_quality', 'Unknown')}
- Overall Risk: {results.reservoir_risk_assessment.get('overall_risk', 'Unknown')}

IDENTIFIED PROSPECTS:
"""
        for prospect in results.prospect_summary:
            interpretation += f"""
- {prospect['prospect_id']} ({prospect['type']}):
  Reservoir: {prospect['reservoir']}
  Seal: {prospect['seal']}
  Risk: {prospect['risk']}
  Notes: {prospect['notes']}
"""

        return interpretation

    def _generate_recommendations(self, results: IntegratedResults) -> str:
        """Generate recommendations based on results"""

        recommendations = """
RECOMMENDATIONS FOR FURTHER ANALYSIS
====================================

1. STRUCTURAL ANALYSIS:
   - Validate fault interpretations with coherence/discontinuity attributes
   - Perform fault seal analysis for identified fault-bounded prospects
   - Map fault timing relative to hydrocarbon charge

2. STRATIGRAPHIC ANALYSIS:
   - Tie facies classification to well data for calibration
   - Map reservoir facies distribution with thickness maps
   - Identify sequence boundaries from facies contacts

3. PROSPECT EVALUATION:
   - Confirm structural closures from horizon interpretation
   - Calculate volumetrics for identified prospects
   - Perform risk assessment with calibrated probabilities

4. DATA INTEGRATION:
   - Integrate DL results with conventional attribute analysis
   - Compare DL facies with petrophysical facies from wells
   - Validate fault detection against manual interpretation

5. MODEL IMPROVEMENT:
   - Fine-tune models with basin-specific training data if available
   - Test multiple architectures for optimal results
   - Consider uncertainty quantification for DL predictions
"""

        return recommendations

    def _create_combined_figure(self, results: IntegratedResults) -> str:
        """Create combined visualization of all results"""

        if not MATPLOTLIB_AVAILABLE or self.seismic_data is None:
            return ""

        output_dir = Path(self.config.output_dir)
        fig_path = output_dir / "combined_interpretation.png"

        nz, ny, nx = self.seismic_data.shape
        inline_idx = ny // 2

        # Create figure
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

        # Plot 1: Seismic
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(self.seismic_data[:, inline_idx, :].T, cmap='gray', aspect='auto')
        ax1.set_title('Seismic Amplitude')
        ax1.set_xlabel('Crossline')
        ax1.set_ylabel('Time (samples)')

        # Plot 2: Fault probability
        if results.fault_results:
            ax2 = fig.add_subplot(gs[0, 1])
            try:
                fault_prob = np.load(output_dir / "faults" / "fault_probability.npy")
                ax2.imshow(fault_prob[:, inline_idx, :].T, cmap='hot', aspect='auto')
                ax2.set_title('Fault Probability')
                ax2.set_xlabel('Crossline')
            except:
                ax2.text(0.5, 0.5, 'Fault data not available', ha='center', va='center')

        # Plot 3: Facies
        if results.facies_results:
            ax3 = fig.add_subplot(gs[0, 2])
            try:
                facies = np.load(output_dir / "facies" / "facies_classification.npy")
                ax3.imshow(facies[:, inline_idx, :].T, cmap='viridis', aspect='auto')
                ax3.set_title('Facies Classification')
                ax3.set_xlabel('Crossline')
            except:
                ax3.text(0.5, 0.5, 'Facies data not available', ha='center', va='center')

        # Plot 4: Combined overlay
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.imshow(self.seismic_data[:, inline_idx, :].T, cmap='gray', aspect='auto')
        try:
            fault_prob = np.load(output_dir / "faults" / "fault_probability.npy")
            ax4.imshow(fault_prob[:, inline_idx, :].T, cmap='Reds', aspect='auto',
                       alpha=0.5, vmin=0, vmax=1)
        except:
            pass
        ax4.set_title('Seismic + Faults Overlay')
        ax4.set_xlabel('Crossline')
        ax4.set_ylabel('Time (samples)')

        # Plot 5: Risk summary
        ax5 = fig.add_subplot(gs[1, 1])
        risk = results.reservoir_risk_assessment
        risk_text = "\n".join([f"{k}: {v}" for k, v in risk.items()])
        ax5.text(0.1, 0.5, f"RESERVOIR RISK\n\n{risk_text}",
                 fontsize=12, family='monospace', va='center')
        ax5.axis('off')
        ax5.set_title('Risk Assessment')

        # Plot 6: Statistics
        ax6 = fig.add_subplot(gs[1, 2])
        stats_text = f"""
DL INTERPRETATION SUMMARY

Faults Detected: {results.fault_results.num_faults_detected if results.fault_results else 'N/A'}
Fault Volume: {results.fault_results.fault_volume_percent:.2f}% if results.fault_results else 'N/A'

Facies Classes: {results.facies_results.num_classes_found if results.facies_results else 'N/A'}

Processing Time: {results.total_processing_time:.1f}s
"""
        ax6.text(0.1, 0.5, stats_text, fontsize=10, family='monospace', va='center')
        ax6.axis('off')
        ax6.set_title('Summary Statistics')

        plt.suptitle(f'Deep Learning Seismic Interpretation - {self.config.basin_name}',
                     fontsize=14, fontweight='bold')

        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()

        return str(fig_path)

    def _generate_report(self, results: IntegratedResults) -> str:
        """Generate comprehensive interpretation report"""

        output_dir = Path(self.config.output_dir)
        report_path = output_dir / f"interpretation_report.{self.config.report_format}"

        report = f"""# Deep Learning Seismic Interpretation Report

## {self.config.basin_name}

**Generated:** {results.timestamp}
**Processing Time:** {results.total_processing_time:.1f} seconds

---

## Executive Summary

This report presents the results of deep learning-based seismic interpretation
using state-of-the-art neural network models for fault detection and facies
classification.

---

## 1. Fault Detection Results

"""
        if results.fault_results:
            report += f"""
- **Model Used:** {results.fault_results.model_used}
- **Faults Detected:** {results.fault_results.num_faults_detected}
- **Fault Volume:** {results.fault_results.fault_volume_percent:.2f}%

### Major Fault Systems

"""
            for fault in results.fault_results.major_fault_systems:
                report += f"- Fault #{fault['rank']}: {fault['size_voxels']:,} voxels, "
                report += f"probability {fault['mean_probability']:.3f}\n"

        report += """
---

## 2. Facies Classification Results

"""
        if results.facies_results:
            report += f"""
- **Model Used:** {results.facies_results.model_used}
- **Classes Identified:** {results.facies_results.num_classes_found}

### Facies Distribution

| Facies | Percentage | Volume (voxels) |
|--------|------------|-----------------|
"""
            for name, pct in sorted(results.facies_results.class_percentages.items(),
                                    key=lambda x: x[1], reverse=True):
                vol = results.facies_results.class_volumes.get(name, 0)
                report += f"| {name} | {pct:.1f}% | {vol:,} |\n"

        report += f"""
---

## 3. Integrated Analysis

### Fault-Facies Correlation

"""
        for facies, data in results.fault_facies_correlation.items():
            report += f"- **{facies}:** {data.get('fault_percentage', 0):.2f}% faulted\n"

        report += f"""
### Reservoir Risk Assessment

| Risk Factor | Assessment |
|-------------|------------|
| Compartmentalization | {results.reservoir_risk_assessment.get('compartmentalization_risk', 'Unknown')} |
| Seal Integrity | {results.reservoir_risk_assessment.get('seal_integrity', 'Unknown')} |
| Reservoir Quality | {results.reservoir_risk_assessment.get('reservoir_quality', 'Unknown')} |
| **Overall Risk** | **{results.reservoir_risk_assessment.get('overall_risk', 'Unknown')}** |

---

## 4. Geological Interpretation

{results.llm_geological_interpretation}

---

## 5. Recommendations

{results.llm_recommendations}

---

## 6. Output Files

- Fault probability volume: `faults/fault_probability.npy`
- Fault binary volume: `faults/fault_binary.npy`
- Facies classification: `facies/facies_classification.npy`
- Combined figure: `combined_interpretation.png`

---

*Report generated by Deep Learning Integration Module*
*PhD Research - Moses Ekene Obasi, University of Calabar*
"""

        with open(report_path, 'w') as f:
            f.write(report)

        return str(report_path)


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """Command-line interface for integrated DL interpretation"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Deep Learning Seismic Interpretation Suite'
    )
    parser.add_argument('segy_file', help='Input SEGY file')
    parser.add_argument('--output-dir', default='dl_interpretation',
                        help='Output directory')
    parser.add_argument('--no-faults', action='store_true',
                        help='Skip fault detection')
    parser.add_argument('--no-facies', action='store_true',
                        help='Skip facies classification')
    parser.add_argument('--no-gpu', action='store_true',
                        help='Disable GPU acceleration')
    parser.add_argument('--no-llm', action='store_true',
                        help='Disable LLM interpretation')

    args = parser.parse_args()

    config = DeepLearningConfig()
    config.seismic_file = args.segy_file
    config.output_dir = args.output_dir
    config.run_fault_detection = not args.no_faults
    config.run_facies_classification = not args.no_facies
    config.use_gpu = not args.no_gpu
    config.enable_llm_interpretation = not args.no_llm

    integration = DeepLearningIntegration(config)
    results = integration.run_full_interpretation()

    print("\n" + "="*70)
    print("INTERPRETATION COMPLETE")
    print("="*70)
    print(f"\nReport: {results.report_file}")
    print(f"Figure: {results.combined_figure}")


if __name__ == '__main__':
    main()
