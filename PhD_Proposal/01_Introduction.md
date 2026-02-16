# AI-Augmented Seismic Interpretation and Reservoir Characterization of the Bornu Chad Basin, Nigeria: Development and Application of an LLM-Assisted Workflow

---

# CHAPTER ONE: INTRODUCTION

## 1.1 Background of Study

The global demand for hydrocarbon resources continues to drive exploration activities into frontier and underexplored sedimentary basins. Among these, the Bornu Chad Basin in northeastern Nigeria represents one of the most promising yet understudied petroliferous provinces in West Africa. Covering approximately 230,000 km² within Nigerian territory and extending into Niger, Chad, and Cameroon, this intracratonic rift basin forms part of the larger West and Central African Rift System (WCARS) that developed during the Early Cretaceous breakup of Gondwana (Genik, 1992; Fairhead et al., 2013).

The petroleum potential of the Bornu Chad Basin has been recognized since the 1970s when exploration activities by the Nigerian National Petroleum Corporation (NNPC) and various international oil companies resulted in the drilling of over 20 exploration wells. These efforts confirmed the presence of a working petroleum system, with oil and gas shows encountered in multiple wells including BULTE-1, HERWA-01, KASADE-01, MASU-1, and others (Avbovbo et al., 1986; Okosun, 1995). Despite these encouraging results, the basin remains largely underexplored compared to the prolific Niger Delta, with no commercial production to date.

The stratigraphic succession of the Bornu Chad Basin comprises four major lithostratigraphic units of petroleum significance: the continental Bima Formation (Albian-Aptian), the transitional Gongila Formation (Cenomanian), the marine Fika Shale (Turonian), and the overlying Chad Formation (Pliocene-Recent). The Bima Sandstone serves as the primary reservoir target, characterized by fluvial to alluvial fan deposits with porosities ranging from 15-28% and permeabilities of 100-2000 mD (Avbovbo et al., 1986). The Fika Shale functions as both the principal source rock, with Total Organic Carbon (TOC) values of 2-5% and Type II/III kerogen, and as the regional seal.

Seismic interpretation remains the cornerstone methodology for subsurface characterization and prospect identification in hydrocarbon exploration. However, traditional seismic interpretation workflows are often time-consuming, subjective, and limited in their ability to extract quantitative information from large datasets. The advent of computational geophysics, machine learning, and automated interpretation techniques presents an opportunity to enhance the efficiency, consistency, and objectivity of seismic interpretation while extracting maximum value from available data (Chopra and Marfurt, 2007; Wrona et al., 2018).

Recent advances in Artificial Intelligence (AI), particularly Large Language Models (LLMs), have revolutionized numerous scientific domains but remain largely unexplored in seismic interpretation workflows. While machine learning has been applied to specific tasks such as fault detection and facies classification, no existing commercial or open-source interpretation system integrates an LLM capable of analyzing results, interpreting both numerical outputs and visual displays, and providing intelligent guidance on interpretation decisions. This represents a significant opportunity to develop a next-generation interpretation framework that combines human expertise with AI-assisted decision support.

This research proposes an integrated seismic interpretation and reservoir characterization study of the Bornu Chad Basin utilizing modern computational approaches combined with conventional geophysical methods, implemented through a novel LLM-assisted interpretation workflow. By leveraging available 3D seismic data, 2D seismic lines, and well log information, this study aims to provide a comprehensive structural and stratigraphic framework that will advance understanding of the basin's petroleum prospectivity and contribute to Nigeria's energy security objectives.

## 1.2 Statement of the Problem

Despite over four decades of exploration activity, the Bornu Chad Basin remains commercially unproductive, presenting a significant gap between recognized petroleum potential and realized hydrocarbon production. Several critical challenges have hindered exploration success and continue to impede understanding of the basin's petroleum systems:

**First**, the structural framework of the basin remains inadequately defined. The complex rift architecture, characterized by multiple half-graben systems bounded by normal faults, creates compartmentalized structural domains that are difficult to correlate across the basin. Previous interpretations have relied primarily on sparse 2D seismic data, resulting in significant uncertainty in fault geometries, displacement distributions, and structural closure delineation.

**Second**, well-to-seismic correlation in the basin has historically been problematic due to limited bandwidth of vintage seismic data (typically 8-15 Hz), resulting in vertical resolution limitations of 50-60 meters. This resolution is often insufficient to resolve individual reservoir intervals within the Bima and Gongila formations, leading to uncertainty in horizon picks and stratigraphic correlations.

**Third**, the identification and evaluation of Direct Hydrocarbon Indicators (DHIs) has not been systematically undertaken. While bright spots and other amplitude anomalies have been observed, their relationship to structural position, reservoir quality, and fluid content remains poorly understood. The absence of quantitative DHI analysis limits the ability to high-grade prospects and reduce exploration risk.

**Fourth**, conventional interpretation workflows applied to the basin have been largely qualitative, producing maps and cross-sections without associated uncertainty estimates or quantitative metrics. This lack of quantification makes it difficult to rank prospects objectively or to communicate confidence levels to decision-makers.

**Fifth**, the integration of well log data with seismic interpretation has been limited. Petrophysical analyses and synthetic seismograms have not been systematically generated or validated, creating a disconnect between borehole observations and seismic response.

**Sixth**, existing interpretation software packages—both commercial (Petrel, Kingdom, OpendTect) and open-source—lack intelligent decision support capabilities. Interpreters must manually evaluate results, decide on next steps, and integrate findings from multiple analysis types without automated guidance. There is no system that can analyze interpretation outputs (numerical and visual), understand the geological context, and recommend optimal interpretation pathways. This gap between data availability and intelligent interpretation support limits the efficiency and consistency of exploration workflows.

These challenges collectively result in high exploration risk and have contributed to the basin's continued status as a frontier province despite its recognized potential. A comprehensive, quantitative interpretation study utilizing modern computational methods is required to address these deficiencies and unlock the basin's hydrocarbon resources.

## 1.3 Justification of the Study

This research is justified on multiple grounds spanning scientific, economic, and methodological considerations:

### Scientific Justification

The Bornu Chad Basin occupies a unique tectonic position at the intersection of multiple rift systems and represents an analogue for other underexplored intracratonic basins in Africa and globally. Understanding the structural evolution, sedimentary fill patterns, and petroleum system dynamics of this basin will contribute to broader knowledge of rift basin petroleum geology. The research will generate new insights into:

- The relationship between rift architecture and hydrocarbon trap formation
- Source rock maturation and migration pathways in continental rift settings
- Reservoir distribution patterns in fluvial-alluvial depositional systems
- The applicability of seismic attributes for hydrocarbon detection in low-frequency seismic data

### Economic Justification

Nigeria's economy remains heavily dependent on hydrocarbon revenues, with petroleum accounting for approximately 90% of export earnings and 60% of government revenues. However, production from the mature Niger Delta province is declining, necessitating exploration of frontier basins to maintain production levels and energy security. The Bornu Chad Basin, with estimated prospective resources of several billion barrels of oil equivalent, represents a strategic target for diversification of Nigeria's hydrocarbon portfolio.

Furthermore, commercial success in the Nigerian portion of the basin could catalyze regional exploration across the international boundaries into Niger, Chad, and Cameroon, with significant implications for West African energy development. The economic multiplier effects of successful exploration—including infrastructure development, employment generation, and technology transfer—would benefit the economically disadvantaged northeastern region of Nigeria.

### Methodological Justification

Traditional seismic interpretation methods, while foundational, are increasingly inadequate for extracting maximum value from modern seismic datasets. This research will demonstrate the application of computational approaches including:

- Automated horizon tracking with quantitative coverage metrics
- Statistical amplitude analysis for DHI identification
- Algorithmic fault detection and characterization
- Integrated well-seismic correlation with correlation coefficient quantification

By developing and validating these methodologies in the context of a real exploration dataset, this research will contribute to the advancement of interpretation practice and provide templates applicable to other frontier basins.

### Technological Innovation Justification

Perhaps most significantly, this research will pioneer the development of an LLM-assisted seismic interpretation workflow—a capability not present in any existing interpretation software. The integration of Large Language Models into the interpretation workflow enables:

- **Intelligent result analysis**: The LLM can interpret numerical outputs, statistical summaries, and quality metrics, providing contextual understanding of results
- **Visual interpretation**: Unlike traditional software, the LLM can analyze generated maps, cross-sections, and attribute displays, identifying patterns and anomalies
- **Decision guidance**: Based on current results and geological knowledge, the LLM can recommend optimal next steps in the interpretation workflow
- **Natural language interaction**: Interpreters can query the system using natural language, making advanced interpretation accessible to users with varying technical backgrounds
- **Integrated documentation**: The LLM automatically documents interpretation decisions, rationale, and results, ensuring reproducibility

This represents a paradigm shift from passive interpretation tools to active interpretation assistants, positioning this research at the forefront of AI-augmented geoscience.

## 1.4 Scope of the Study

This research encompasses the following spatial, temporal, and methodological boundaries:

### Spatial Scope

The study area covers approximately 1,955 km² of the central Bornu Chad Basin as defined by the available 3D seismic survey. The survey extends from Inline 5047 to 6047 and Crossline 4885 to 7020, encompassing approximately 1.84 million seismic traces. Additionally, 62 2D seismic lines from the CH-78 survey (vintage 1978) will be incorporated for regional context. Six wells with comprehensive log suites (BULTE-1, HERWA-01, KASADE-01, MASU-1, NGAMMAEAST-1, NGORNORTH-1) will provide subsurface calibration, with three wells (BULTE-1, HERWA-01, KASADE-01) located within the 3D seismic coverage.

### Stratigraphic Scope

The interpretation will focus on the Cretaceous petroleum system, specifically:
- Top Chad Formation (~400-600 ms TWT)
- Top Fika Shale (~900-1100 ms TWT) - Source rock and seal
- Top Gongila Formation (~1100-1300 ms TWT) - Secondary reservoir
- Top Bima Sandstone (~1500-1700 ms TWT) - Primary reservoir target
- Basement (where imageable)

### Methodological Scope

The research will employ:
- Seismic data quality assessment and conditioning
- Well log analysis and petrophysical evaluation
- Synthetic seismogram generation and well-seismic correlation
- Horizon interpretation and structural mapping
- Seismic attribute extraction and analysis
- Fault interpretation and characterization
- Volumetric estimation and prospect ranking

The study will not include seismic reprocessing, acquisition of new data, or drilling operations.

## 1.5 Aim and Objectives

### Aim

The aim of this research is to conduct a comprehensive seismic interpretation and reservoir characterization of the Bornu Chad Basin, Nigeria, utilizing integrated computational and conventional geophysical methods to define the structural framework, identify hydrocarbon prospects, and quantify exploration potential.

### Objectives

To achieve this aim, the following specific objectives will be pursued:

1. **To assess the quality of available seismic and well data** through systematic evaluation of data completeness, frequency content, signal-to-noise ratio, and spatial coverage, thereby establishing the interpretational constraints and resolution limits.

2. **To establish well-to-seismic correlation** through generation of synthetic seismograms from sonic and density logs, enabling calibration of seismic reflectors to stratigraphic horizons with quantified correlation metrics.

3. **To map the major stratigraphic horizons** (Top Chad, Top Fika, Top Gongila, Top Bima) across the 3D seismic volume using a combination of manual and automated tracking techniques, producing time-structure maps with full survey coverage.

4. **To identify and characterize the fault network** through variance attribute analysis and gradient-based fault detection, quantifying fault geometries, throws, and orientations to establish the structural framework.

5. **To extract and analyze seismic attributes** including RMS amplitude, for identification of potential Direct Hydrocarbon Indicators (DHIs) and reservoir quality variations.

6. **To identify structural and stratigraphic trapping configurations** through closure analysis and integration of structural and amplitude information.

7. **To estimate hydrocarbon volumes** for identified prospects using deterministic and probabilistic methods, providing risked resource assessments.

8. **To develop exploration recommendations** based on integrated interpretation results, ranking prospects by geological risk and resource potential.

9. **To design and implement an LLM-assisted interpretation workflow** with graphical user interface (GUI) that integrates all interpretation modules, enables intelligent analysis of results (numerical and visual), and provides decision support for interpretation guidance.

## 1.6 Research Questions

This research will address the following questions:

1. What is the structural style and fault architecture of the study area, and how do these control potential hydrocarbon trapping configurations?

2. What is the quality of well-to-seismic ties achievable with available data, and what are the implications for interpretation confidence?

3. What is the distribution and character of potential reservoir intervals as expressed in seismic amplitude response?

4. Where are the most prospective locations for hydrocarbon accumulation based on integrated structural and stratigraphic analysis?

5. What are the estimated hydrocarbon volumes and associated uncertainties for identified prospects?

## 1.7 Significance of the Study

This research will make significant contributions in several domains:

**To the petroleum industry**: The study will provide the most comprehensive publicly available interpretation of the Bornu Chad Basin 3D seismic data, generating prospect inventories and resource estimates that can guide future exploration investment.

**To academic knowledge**: The research will advance understanding of rift basin petroleum systems and demonstrate the application of modern computational interpretation methods to frontier exploration datasets.

**To methodology development**: The integrated workflow developed in this research, combining automated interpretation techniques with conventional methods, will serve as a template for similar studies in data-limited frontier basins.

**To national development**: By reducing exploration risk and identifying drilling targets, this research will contribute to Nigeria's strategic objective of diversifying hydrocarbon production beyond the Niger Delta.

