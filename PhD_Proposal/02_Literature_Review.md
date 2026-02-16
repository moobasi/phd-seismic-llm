# CHAPTER TWO: LITERATURE REVIEW

## 2.1 Regional Geological Setting

### 2.1.1 Tectonic Framework

The Bornu Chad Basin forms part of the West and Central African Rift System (WCARS), a series of Cretaceous extensional basins that developed in response to the separation of South America from Africa during the breakup of Gondwana (Fairhead, 1986; Genik, 1992). The basin is bounded to the southwest by the Precambrian basement of the Nigerian Shield, to the southeast by the Benue Trough, and extends northward into the Republic of Niger, Chad, and Cameroon.

Genik (1993) recognized four major tectonic phases in the evolution of WCARS basins:
1. **Early Cretaceous rifting (140-118 Ma)**: Initial continental extension with development of NW-SE trending half-grabens
2. **Late Aptian-Cenomanian thermal subsidence (118-95 Ma)**: Post-rift sag phase with regional subsidence
3. **Santonian compression (84 Ma)**: Basin inversion and reactivation of basement structures
4. **Late Cretaceous-Tertiary thermal subsidence (84-0 Ma)**: Renewed subsidence and sediment accumulation

The structural style is dominated by normal faults trending NE-SW and NW-SE, consistent with regional extension directions. Fairhead et al. (2013) utilized gravity and magnetic data to delineate major basement structures, identifying multiple sub-basins separated by intra-basinal highs. The Maiduguri Trough, where the study area is located, represents one of the deepest depocenters with sedimentary thicknesses exceeding 5 km.

### 2.1.2 Stratigraphic Framework

The stratigraphic succession of the Bornu Chad Basin has been described by numerous workers (Carter et al., 1963; Avbovbo et al., 1986; Okosun, 1995; Obaje, 2009). The sedimentary fill comprises:

**Bima Formation (Albian-Aptian)**: The basal unit consists of continental clastic sediments deposited in fluvial to alluvial fan environments. Lithologically, it comprises coarse-grained, poorly sorted sandstones with conglomeratic intervals, interbedded with red and mottled claystones. Thickness varies from 100 m on basin margins to over 3000 m in depocenters. The Bima Sandstone is the primary reservoir target in the basin.

**Gongila Formation (Cenomanian)**: Transitional marine sediments comprising interbedded sandstones, shales, and limestones. The formation records the initial marine transgression into the basin. Reservoir quality is variable, with net-to-gross ratios typically 0.3-0.5.

**Fika Shale (Turonian)**: Marine shales with minor limestone interbeds, deposited during maximum marine transgression. This unit serves dual functions as the principal source rock (TOC 2-5%, Type II/III kerogen) and regional seal. Thickness ranges from 300-500 m.

**Chad Formation (Pliocene-Recent)**: Lacustrine and fluvial sediments associated with the Lake Chad system. Not prospective for hydrocarbons but important for near-surface characterization.

### 2.1.3 Petroleum System Elements

The petroleum system of the Bornu Chad Basin has been characterized by Obaje et al. (2004) and Adepelumi et al. (2011):

**Source Rock**: The Fika Shale is the proven source rock, with geochemical analyses indicating TOC values of 2-5% and Hydrogen Index values of 150-350 mg HC/g TOC. Thermal maturity studies suggest the Fika is within the oil window (Ro 0.6-0.8%) in deeper parts of the basin, transitioning to gas window at greater depths.

**Reservoir**: The Bima Sandstone provides the primary reservoir, with measured porosities of 15-28% and permeabilities of 100-2000 mD from well data. Secondary reservoir potential exists in the Gongila Formation.

**Seal**: The Fika Shale provides regional top seal. Fault seal is critical for fault-bounded traps and depends on shale gouge ratio and fault rock properties.

**Trap**: Both structural and stratigraphic traps are possible. Structural traps include fault-bounded closures, rollover anticlines in hanging walls, and horst blocks. Stratigraphic traps may occur where Bima sands pinch out against basement highs or through facies changes.

**Migration**: Migration pathways are interpreted to be primarily along fault planes and carrier beds within the Bima Formation.

## 2.2 Previous Exploration and Research

### 2.2.1 Exploration History

Petroleum exploration in the Bornu Chad Basin began in 1959 with gravity and magnetic surveys conducted by the Geological Survey of Nigeria. The first exploration well, Wadi-1, was drilled in 1959 but found no significant shows (Avbovbo et al., 1986).

Intensive exploration during the 1970s and 1980s by NNPC and Shell resulted in the drilling of over 20 wells. Significant hydrocarbon shows were encountered in several wells:
- **MASU-1**: Gas shows in Bima Sandstone
- **BULTE-1**: Oil and gas shows
- **HERWA-01**: Gas shows with pressure data
- **KASADE-01**: Oil shows in multiple intervals

Despite these encouraging results, no commercial discoveries were declared due to a combination of factors including small accumulation sizes, lack of infrastructure, and focus on the more prolific Niger Delta.

### 2.2.2 Previous Seismic Interpretation Studies

Seismic interpretation studies in the Bornu Chad Basin have been limited compared to other Nigerian basins. Key contributions include:

Okosun (1995) conducted a regional stratigraphic and structural analysis using available 2D seismic and well data, identifying the major formation tops and delineating basement structure. This work established the fundamental stratigraphic framework but lacked quantitative depth conversion due to limited velocity control.

Ola-Buraimo et al. (2017) examined the biostratigraphy and sequence stratigraphy of the basin, correlating well logs to establish chronostratigraphic surfaces. Their work provided improved age dating of formations but did not include seismic interpretation.

Adepelumi et al. (2011) applied potential field methods (gravity and magnetics) integrated with limited seismic to model basement structure and sediment thickness. Their results indicated sedimentary thicknesses exceeding 7 km in the deepest depocenters.

None of these studies have utilized the available 3D seismic data for comprehensive structural and stratigraphic interpretation with modern computational methods.

## 2.3 Seismic Interpretation Methods

### 2.3.1 Well-to-Seismic Correlation

Well-to-seismic tie is fundamental to calibrating seismic interpretation with subsurface geology. The standard approach involves:

1. **Synthetic seismogram generation**: Calculating acoustic impedance (AI = velocity × density) from sonic and density logs, deriving reflection coefficients, and convolving with an extracted or assumed wavelet (Sheriff and Geldart, 1995).

2. **Correlation optimization**: Adjusting time shift and phase rotation to maximize correlation between synthetic and seismic traces (White and Simm, 2003).

3. **Quality assessment**: Quantifying tie quality through correlation coefficient, with values >0.7 considered excellent, 0.5-0.7 good, 0.3-0.5 fair, and <0.3 poor.

Challenges in well-seismic ties include:
- Wavelet estimation uncertainty
- Well location errors relative to seismic
- Velocity anisotropy and dispersion effects
- Log quality issues (borehole washouts, calibration errors)

For vintage seismic data with limited bandwidth (8-15 Hz), correlation coefficients of 0.3-0.5 are commonly achieved and considered acceptable for regional interpretation (Bacon et al., 2003).

### 2.3.2 Horizon Interpretation

Horizon interpretation has evolved from purely manual picking to include various automated and semi-automated techniques:

**Manual interpretation**: Traditional approach where interpreters pick reflection events on seismic sections. While accurate, it is time-consuming and subject to interpreter bias.

**Auto-tracking**: Algorithms that propagate picks from seed points by following amplitude maxima, minima, or zero crossings within defined search windows (Borgos et al., 2003). Auto-tracking increases efficiency but requires quality control.

**Machine learning approaches**: Recent advances include convolutional neural networks for horizon detection and tracking (Wu and Fomel, 2018), though these require training data that may not be available for frontier basins.

This research will employ a seed-based auto-tracking approach with wells as seed points, combining efficiency with well-calibration.

### 2.3.3 Seismic Attributes

Seismic attributes are quantities derived from seismic data that provide information beyond simple reflector geometry. Key attributes for hydrocarbon detection include:

**Amplitude-based attributes**:
- RMS amplitude: Root-mean-square of amplitudes in a window; sensitive to acoustic impedance contrasts and fluid content
- Maximum amplitude: Peak amplitude value; highlights bright spots
- Average absolute amplitude: Mean of absolute values; smoothed amplitude response

Chopra and Marfurt (2007) provide a comprehensive review of seismic attributes and their geological significance. Brown (2011) specifically addresses amplitude interpretation for DHI analysis.

**Frequency-based attributes**:
- Instantaneous frequency: Related to bed thickness and attenuation
- Spectral decomposition: Frequency-dependent response related to tuning effects

**Structural attributes**:
- Variance/coherence: Measures trace-to-trace similarity; highlights faults and discontinuities
- Curvature: Second derivative of structure; indicates flexure and fracturing

For DHI analysis, amplitude anomalies exceeding two standard deviations from background are commonly used as statistical thresholds for anomaly identification (Roden et al., 2015).

### 2.3.4 Fault Interpretation

Fault interpretation is critical in rift basins where fault-bounded traps are common. Traditional approaches involve manual picking of fault planes from discontinuities in seismic reflectors. Automated approaches include:

**Variance/coherence**: Low coherence values indicate lateral discontinuities associated with faults (Bahorich and Farmer, 1995).

**Semblance**: Similar to coherence but based on waveform similarity.

**Gradient-based methods**: Sharp gradients in horizon surfaces indicate fault locations and can be used to extract fault traces (Admasu et al., 2006).

Fault characterization includes:
- Throw estimation: Vertical displacement across fault
- Strike and dip measurement: Fault orientation
- Length and connectivity: Extent and linkage of fault segments

## 2.4 Reservoir Characterization

### 2.4.1 Petrophysical Analysis

Petrophysical analysis of well logs provides quantitative reservoir properties:

**Shale volume (Vsh)**: Calculated from gamma ray using linear or non-linear responses:
Vsh = (GR - GRclean) / (GRshale - GRclean)

**Porosity**: Derived from density, neutron, or sonic logs:
φ = (ρma - ρb) / (ρma - ρf)

**Water saturation (Sw)**: Calculated using Archie's equation:
Sw = (a × Rw / (φ^m × Rt))^(1/n)

Where typical parameters for clastic reservoirs are: a=1, m=2, n=2.

**Permeability**: Estimated from porosity-permeability transforms calibrated to core data.

### 2.4.2 Volumetric Estimation

Hydrocarbon volumes are estimated using the standard STOIIP equation:
STOIIP = (A × h × NTG × φ × (1-Sw)) / Bo × 6.2898

Where:
- A = area (m²)
- h = net pay thickness (m)
- NTG = net-to-gross ratio
- φ = porosity
- Sw = water saturation
- Bo = formation volume factor

Probabilistic methods (Monte Carlo simulation) allow quantification of uncertainty through P10/P50/P90 estimates.

## 2.5 Artificial Intelligence in Seismic Interpretation

### 2.5.1 Machine Learning Applications

The application of machine learning (ML) to seismic interpretation has grown substantially in recent years. Key developments include:

**Fault Detection**: Convolutional Neural Networks (CNNs) have been applied to automate fault detection from seismic volumes. Wu and Fomel (2018) demonstrated automatic fault interpretation using optimal surface voting, achieving results comparable to manual interpretation. Wrona et al. (2018) applied machine learning for seismic facies analysis, showing that supervised classification can effectively delineate geological features.

**Horizon Tracking**: Automated horizon tracking algorithms have evolved from simple amplitude followers to sophisticated machine learning approaches. Neural network-based methods can learn complex reflection patterns and track horizons through noisy or discontinuous data (Borgos et al., 2003).

**Seismic Facies Classification**: Self-organizing maps (SOMs) and principal component analysis (PCA) have been applied to multi-attribute seismic data for unsupervised facies classification (Roden et al., 2015). These methods can identify geological patterns not apparent in individual attributes.

### 2.5.2 Large Language Models in Geoscience

Large Language Models (LLMs) represent the latest advancement in artificial intelligence, demonstrating remarkable capabilities in understanding context, generating human-like text, and reasoning across domains. However, their application to seismic interpretation remains virtually unexplored.

Current limitations in the field include:

- **No integrated LLM-interpretation systems**: While LLMs have been applied to general scientific literature analysis, no seismic interpretation software incorporates LLM capabilities for result analysis or decision guidance
- **Disconnected workflows**: Existing tools perform individual tasks (fault detection, horizon tracking, attribute analysis) but cannot intelligently synthesize results or recommend next steps
- **Limited multimodal analysis**: Current ML applications process numerical data but cannot interpret visual outputs (maps, sections) in conjunction with quantitative results
- **No natural language interface**: Interpreters must navigate complex software menus rather than expressing interpretation goals in natural language

This represents a significant opportunity for innovation—the development of an LLM-assisted interpretation framework that can understand both numerical and visual outputs, maintain geological context, and provide intelligent guidance throughout the interpretation workflow.

## 2.6 Research Gap

Review of the literature reveals significant gaps that this research will address:

1. **No comprehensive 3D seismic interpretation** of the Bornu Chad Basin has been published utilizing modern computational methods.

2. **Quantitative well-seismic ties** with correlation metrics have not been systematically reported for the basin.

3. **Seismic attribute analysis** for DHI identification has not been applied to the available 3D data.

4. **Automated fault characterization** with throw quantification has not been undertaken.

5. **Integrated structural-stratigraphic interpretation** combining all available data types is lacking.

6. **No LLM-assisted interpretation workflow** exists in the geoscience domain that can analyze both numerical results and visual outputs, understand geological context, and provide intelligent guidance for interpretation decisions. This represents a fundamental gap in the evolution of interpretation technology.

This research will fill these gaps by applying a comprehensive, quantitative interpretation workflow to the available dataset, implemented through a novel LLM-assisted framework with graphical user interface.

## 2.7 Conceptual Framework

The research is underpinned by the following conceptual framework:

**Petroleum systems analysis**: Understanding hydrocarbon occurrence requires integration of source, reservoir, seal, trap, and migration elements (Magoon and Dow, 1994).

**Seismic stratigraphy**: Seismic reflections follow chronostratigraphic surfaces, enabling correlation of depositional sequences (Vail et al., 1977).

**Quantitative seismic interpretation**: Seismic data contains quantitative information about subsurface properties that can be extracted through calibrated attribute analysis (Avseth et al., 2005).

**Integrated interpretation**: Maximum value is extracted when all available data types are integrated in a consistent framework.

**AI-augmented geoscience**: Large Language Models can serve as intelligent assistants that understand domain knowledge, analyze multimodal data (text, numbers, images), and provide contextual guidance—transforming interpretation from a purely manual process to a human-AI collaborative workflow.

