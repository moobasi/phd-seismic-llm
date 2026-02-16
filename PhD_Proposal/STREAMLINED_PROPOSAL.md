# PhD RESEARCH PROPOSAL

## Development of an LLM-Assisted Seismic Interpretation Framework: Validation Using Bornu Chad Basin Data, Nigeria

**Submitted to the Department of Geology, Faculty of Science, University of Calabar, Nigeria**

**In Partial Fulfillment of the Requirements for the Award of Doctor of Philosophy (PhD) in Geology**

**Candidate:** MOSES EKENE OBASI

**Supervisor:** PROF. DOMINIC AKAM OBI

**Department of Geology, Faculty of Science, University of Calabar, Calabar, Nigeria**

---

# CHAPTER ONE: INTRODUCTION

## 1.1 Background to the Study

Seismic interpretation is the cornerstone of hydrocarbon exploration, enabling geoscientists to visualize subsurface structures, identify potential reservoirs, and assess drilling targets (Sheriff & Geldart, 1995). The interpretation process involves analyzing seismic data alongside well logs to understand geological formations, fault systems, and stratigraphic relationships (Brown, 2011). Despite advances in seismic acquisition and processing, the interpretation stage remains largely manual, time-consuming, and dependent on individual expertise (Bacon et al., 2003).

Modern interpretation software packages provide powerful visualization and analysis tools but function as passive tools that execute user commands without understanding geological context or providing intelligent guidance (Chopra & Marfurt, 2007). The emergence of Large Language Models (LLMs) represents a transformative development in artificial intelligence, demonstrating remarkable capabilities in understanding context, reasoning across domains, and generating human-like responses (Brown et al., 2020; LeCun et al., 2015). LLMs have been successfully applied in medicine, law, and scientific research (Anthropic, 2024), yet their application to seismic interpretation remains unexplored.

This research proposes the development of an LLM-assisted seismic interpretation framework that integrates Large Language Model capabilities directly into the interpretation workflow. The framework will be validated using real seismic and well data from the Bornu Chad Basin, Nigeria (Genik, 1992; Obaje, 2009).

## 1.2 Statement of the Problem

Current seismic interpretation workflows face fundamental limitations (Brown, 2011; Chopra & Marfurt, 2007): (1) existing software lacks intelligent decision support, forcing interpreters to evaluate results based solely on experience (Bacon et al., 2003); (2) software cannot analyze visual outputs such as maps and cross-sections (Avseth et al., 2005); (3) there is no natural language interface, creating steep learning curves (Dramsch, 2020); (4) interpretation documentation is manual and often incomplete; and (5) quality assurance relies entirely on human review with no automated error detection (Zhao & Mukhopadhyay, 2018). These limitations result in inefficient, inconsistent, and poorly documented interpretation workflows.

## 1.3 Justification of the Study

**Methodological Justification:** Large Language Models have demonstrated transformative potential across professional domains (Brown et al., 2020; OpenAI, 2023). Seismic interpretation, requiring integration of diverse data types and expert judgment, is well-suited for LLM assistance (Dramsch, 2020).

**Practical Justification:** The petroleum industry faces a growing expertise gap as experienced interpreters retire (Chopra & Marfurt, 2007). An LLM-assisted system could provide junior interpreters with expert-level guidance, ensuring consistent application of best practices.

**Scientific Justification:** This research will demonstrate the integration of multimodal AI capabilities in a geoscience context (LeCun et al., 2015; Zhao & Mukhopadhyay, 2018), providing empirical evidence of the framework's effectiveness and limitations.

## 1.4 Scope of the Study

The research will develop an LLM-assisted interpretation framework with capabilities for analyzing numerical outputs, interpreting visual outputs (maps, cross-sections, attribute displays), providing decision guidance, natural language interaction, and automated documentation. The framework will be validated using seismic and well data from the Bornu Chad Basin, Nigeria, including a 3D seismic survey (~1,955 km²) and well log data from six exploration wells (Genik, 1992, 1993; Obaje, 2009; NNPC, n.d.). The research will not include development of new seismic processing algorithms, acquisition of new data, or real-time drilling applications.

## 1.5 Aim and Objectives

**Aim:** To develop and validate an LLM-assisted seismic interpretation framework that provides intelligent analysis and decision guidance throughout the interpretation workflow.

**Objectives:**
1. To design the architecture of an LLM-assisted interpretation framework integrating Large Language Model capabilities with standard seismic interpretation functions.
2. To implement multimodal analysis capabilities for interpreting seismic maps, cross-sections, and attribute displays.
3. To develop a decision guidance system recommending interpretation steps based on current results and geological context.
4. To validate the framework using real seismic and well data from the Bornu Chad Basin (Genik, 1992, 1993; Obaje, 2009).
5. To evaluate the framework's effectiveness in terms of interpretation efficiency, consistency, and documentation quality.

---

# CHAPTER TWO: LITERATURE REVIEW

## 2.1 Artificial Intelligence in Geoscience

The application of artificial intelligence to geoscience has grown substantially over the past decade (Dramsch, 2020). Machine learning techniques have been successfully applied to fault detection using Convolutional Neural Networks (Wu & Fomel, 2018; Admasu et al., 2006), seismic facies classification using self-organizing maps (Wrona et al., 2018; Roden et al., 2015), and horizon tracking using pattern recognition algorithms (Borgos et al., 2003; Zhao & Mukhopadhyay, 2018).

Despite these advances, current AI applications are narrow in scope, addressing single tasks without understanding the broader interpretation context (Dramsch, 2020). These systems cannot synthesize results from multiple analysis types, recommend appropriate workflow steps, or explain outputs in geological terms (Brown, 2011).

## 2.2 Large Language Models and Their Capabilities

Large Language Models represent a fundamentally different approach to artificial intelligence, trained on vast text corpora to develop broad capabilities in language understanding, reasoning, and knowledge application (Brown et al., 2020; LeCun et al., 2015). Modern LLMs demonstrate contextual understanding, reasoning capabilities, knowledge application, and natural language interaction (Anthropic, 2024; OpenAI, 2023). Recent developments include multimodal models capable of processing both text and images, opening possibilities for AI systems that can interpret geological maps and seismic displays (OpenAI, 2023; Anthropic, 2024).

## 2.3 Gap in Current Technology

Review of current technology reveals a significant gap: no seismic interpretation system integrates LLM capabilities for intelligent analysis and guidance.

| Capability | Commercial Software | LLM-Assisted (This Research) |
|------------|-------------------|------------------------------|
| Seismic display & tracking | Yes | Yes |
| Result interpretation | No | Yes |
| Visual analysis | No | Yes |
| Decision guidance | No | Yes |
| Natural language interface | No | Yes |

## 2.4 Validation Data Context

The framework will be validated using data from the Bornu Chad Basin, an intracratonic rift basin in northeastern Nigeria containing a proven petroleum system (Genik, 1992, 1993; Obaje et al., 2004; Okosun, 1995). The basin developed as part of the West and Central African Rift System during Late Cretaceous extension (Fairhead et al., 2013), characterized by northeast-trending horst and graben structures (Avbovbo et al., 1986). Available data includes a 3D seismic survey of approximately 1,955 square kilometers and well logs from six exploration wells (NNPC, n.d.; DPR, n.d.), providing realistic interpretation challenges (Ola-Buraimo et al., 2017; Magoon & Dow, 1994).

---

# CHAPTER THREE: METHODOLOGY

## 3.1 Research Design

This research employs a design science methodology involving iterative cycles of framework design, implementation, and evaluation (Dramsch, 2020). The research proceeds in three phases: (1) framework design and architecture development, (2) implementation and integration with interpretation functions, and (3) validation using real seismic and well data (Genik, 1992; Obaje, 2009).

## 3.2 Framework Architecture

The LLM-assisted interpretation framework consists of three integrated layers:

**Data Layer:** Handles seismic and well data interactions, including reading/parsing data in standard formats, managing interpretation products, and storing session history.

**Processing Layer:** Implements standard interpretation functions (Brown, 2011; Sheriff & Geldart, 1995) including seismic data quality assessment, well log analysis (Avseth et al., 2005), synthetic seismogram generation (White & Simm, 2003), horizon tracking (Vail et al., 1977), seismic attribute extraction (Chopra & Marfurt, 2007), and fault detection (Bahorich & Farmer, 1995).

**LLM Layer:** Provides intelligent analysis by receiving processing outputs, analyzing results in geological context, interpreting visual outputs, generating workflow recommendations, responding to natural language queries, and documenting interpretation decisions.

## 3.3 LLM Integration Approach

The framework integrates LLM capabilities through: (1) **Result Analysis** providing geological interpretation of numerical summaries and quality indicators; (2) **Visual Interpretation** using multimodal LLM to describe displays, identify geological features, and assess map quality; (3) **Decision Guidance** recommending next steps and parameter adjustments; and (4) **Natural Language Interface** allowing questions about results, geological context, and software functionality.

## 3.4 Implementation

The framework is implemented in Python utilizing NumPy (Harris et al., 2020), SciPy (Virtanen et al., 2020), Matplotlib (Hunter, 2007), segyio (segyio Development Team, 2023), lasio (lasio Development Team, 2023), Tkinter (Python Software Foundation, 2024), and Claude API (Anthropic, 2024). Geological knowledge is incorporated through interpretation best practices (Brown, 2011; Bacon et al., 2003) and study area context (Genik, 1992; Obaje, 2009).

## 3.5 Validation Approach

**Validation Data:** 3D seismic volume (~1,955 km², ~1.8 million traces), 62 2D seismic lines, and well logs from 6 wells (NNPC, n.d.; DPR, n.d.).

**Validation Tasks:** (1) Data quality assessment (Yilmaz, 2001); (2) Well-seismic correlation (White & Simm, 2003); (3) Horizon mapping (Vail et al., 1977); and (4) Fault interpretation (Bahorich & Farmer, 1995; Wu & Fomel, 2018; Fairhead et al., 2013).

**Evaluation Criteria:** Accuracy (comparison with expert evaluation), usefulness (guidance quality), efficiency (time comparison), and documentation quality.

---

# CHAPTER FOUR: EXPECTED CONTRIBUTIONS

## 4.1 Primary Contribution

The primary contribution is the development and validation of the first LLM-assisted seismic interpretation framework, representing a paradigm shift from passive tools to active interpretation assistants (Dramsch, 2020; Zhao & Mukhopadhyay, 2018). Novel capabilities include intelligent result analysis (Chopra & Marfurt, 2007), visual interpretation (OpenAI, 2023), decision guidance (Brown, 2011), natural language interface (Anthropic, 2024), and automated documentation.

This contribution demonstrates the first integration of LLM capabilities into seismic interpretation, provides a template for AI-assisted geological analysis, addresses the expertise gap in the petroleum industry, and improves interpretation consistency (LeCun et al., 2015; Dramsch, 2020).

## 4.2 Methodological Contributions

The research contributes: (1) documented architecture for integrating LLMs with interpretation software; (2) methods for LLM interpretation of geological visualizations; and (3) a framework for evaluating LLM-assisted interpretation including accuracy criteria and documentation metrics.

## 4.3 Applied Contributions

The research produces a functional interpretation system and interpretation products for the Bornu Chad Basin including horizon maps (Vail et al., 1977), fault characterization (Fairhead et al., 2013), and attribute analysis results (Chopra & Marfurt, 2007), contributing to basin understanding (Adepelumi et al., 2011; Okosun, 1995).

---

# CHAPTER FIVE: SUMMARY AND CONCLUSION

## 5.1 Summary

This proposal presents a research plan for developing and validating an LLM-assisted seismic interpretation framework addressing the absence of intelligent systems that can analyze results, interpret visual outputs, and provide guidance throughout interpretation workflows (Dramsch, 2020; Zhao & Mukhopadhyay, 2018). The framework integrates Large Language Model capabilities (Brown et al., 2020; Anthropic, 2024) and will be validated using Bornu Chad Basin data (Genik, 1992; Obaje, 2009).

## 5.2 Research Justification

This research merits doctoral-level investigation based on: **Originality** (first integration of LLM capabilities into seismic interpretation); **Significance** (addresses expertise gaps and interpretation inconsistency); **Rigor** (systematic methodology with real geological data validation); and **Contribution** (working software system and documented methods).

## 5.3 Expected Outcomes

1. LLM-assisted interpretation framework with documented architecture
2. Validation results demonstrating framework performance
3. Methodological documentation for LLM integration in geological applications
4. Interpretation products for the Bornu Chad Basin study area

## 5.4 Conclusion

This research addresses a timely opportunity to integrate artificial intelligence advances with seismic interpretation practice. The proposed framework represents a new paradigm where intelligent systems assist human experts rather than simply executing commands. The combination of novel AI integration, practical validation, and documented methodology positions this research to make significant contributions to both geoscience and artificial intelligence applications.

---

# REFERENCES

Adepelumi, A. A., Alao, O. A., & Kutemi, T. F. (2011). Integrated geophysical mapping of the basement structures and correlation with hydrocarbon trap formation in the Bornu basin, Nigeria. *Journal of African Earth Sciences*, *61*(4), 311–320. https://doi.org/10.1016/j.jafrearsci.2011.08.004

Admasu, F., Back, S., & Toennies, K. (2006). Autotracking of faults on 3D seismic data. *Geophysics*, *71*(4), A49–A53. https://doi.org/10.1190/1.2215868

Anthropic. (2024). *Claude: A family of highly capable AI assistants* [Technical documentation]. https://www.anthropic.com

Avseth, P., Mukerji, T., & Mavko, G. (2005). *Quantitative seismic interpretation: Applying rock physics tools to reduce interpretation risk*. Cambridge University Press.

Avbovbo, A. A., Ayoola, E. O., & Osahon, G. A. (1986). Depositional and structural styles in the Chad Basin of northeastern Nigeria. *AAPG Bulletin*, *70*(12), 1787–1798.

Bacon, M., Simm, R., & Redshaw, T. (2003). *3-D seismic interpretation*. Cambridge University Press.

Bahorich, M., & Farmer, S. (1995). 3-D seismic discontinuity for faults and stratigraphic features: The coherence cube. *The Leading Edge*, *14*(10), 1053–1058. https://doi.org/10.1190/1.1437077

Borgos, H. G., Skov, T., Randen, T., & Sonneland, L. (2003). Automated geometry extraction from 3D seismic data. *SEG Technical Program Expanded Abstracts*, *22*, 1541–1544.

Brown, A. R. (2011). *Interpretation of three-dimensional seismic data* (7th ed.). AAPG Memoir 42.

Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., Agarwal, S., Herbert-Voss, A., Krueger, G., Henighan, T., Child, R., Ramesh, A., Ziegler, D. M., Wu, J., Winter, C., ... Amodei, D. (2020). Language models are few-shot learners. *Advances in Neural Information Processing Systems*, *33*, 1877–1901.

Chopra, S., & Marfurt, K. J. (2007). *Seismic attributes for prospect identification and reservoir characterization*. SEG Geophysical Developments Series No. 11.

Department of Petroleum Resources. (n.d.). *3D seismic survey data release documentation, Bornu Chad Basin* [Data release documentation]. DPR, Nigeria.

Dramsch, J. S. (2020). 70 years of machine learning in geoscience in review. *Advances in Geophysics*, *61*, 1–55. https://doi.org/10.1016/bs.agph.2020.08.002

Fairhead, J. D., Green, C. M., Masterton, S. M., & Guiraud, R. (2013). The role that plate tectonics, inferred stress changes and stratigraphic unconformities have on the evolution of the West and Central African Rift System. *Tectonophysics*, *594*, 118–127. https://doi.org/10.1016/j.tecto.2013.03.021

Genik, G. J. (1992). Regional framework, structural and petroleum aspects of rift basins in Niger, Chad and the Central African Republic. *Tectonophysics*, *213*(1–2), 169–185. https://doi.org/10.1016/0040-1951(92)90257-7

Genik, G. J. (1993). Petroleum geology of Cretaceous–Tertiary rift basins in Niger, Chad, and Central African Republic. *AAPG Bulletin*, *77*(8), 1405–1434.

Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., Virtanen, P., Cournapeau, D., ... Oliphant, T. E. (2020). Array programming with NumPy. *Nature*, *585*, 357–362. https://doi.org/10.1038/s41586-020-2649-2

Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. *Computing in Science & Engineering*, *9*(3), 90–95.

lasio Development Team. (2023). *lasio: Python library for reading and writing LAS files* [Computer software]. https://github.com/kinverarity1/lasio

LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature*, *521*(7553), 436–444. https://doi.org/10.1038/nature14539

Magoon, L. B., & Dow, W. G. (1994). The petroleum system. In L. B. Magoon & W. G. Dow (Eds.), *The petroleum system—From source to trap* (pp. 3–24). AAPG Memoir 60.

Nigerian National Petroleum Corporation. (n.d.). *Well completion reports for Bornu Chad Basin exploration wells* [Restricted technical reports]. NNPC.

Obaje, N. G. (2009). *Geology and mineral resources of Nigeria* (Lecture Notes in Earth Sciences 120). Springer-Verlag.

Obaje, N. G., Wehner, H., Scheeder, G., Abubakar, M. B., & Jauro, A. (2004). Hydrocarbon prospectivity of Nigeria's inland basins. *AAPG Bulletin*, *88*(3), 325–353.

Okosun, E. A. (1995). Review of the geology of Bornu Basin. *Journal of Mining and Geology*, *31*(2), 113–122.

Ola-Buraimo, A. O., Akaegbobi, I. M., & Ogungbesan, G. O. (2017). Palynological and paleoenvironmental investigation of the Fika Shale, Borno Basin, Northeastern Nigeria. *Journal of African Earth Sciences*, *135*, 32–51.

OpenAI. (2023). *GPT-4 technical report* (arXiv:2303.08774). arXiv. https://arxiv.org/abs/2303.08774

Python Software Foundation. (2024). *Tkinter—Python interface to Tcl/Tk* [Documentation]. https://docs.python.org/3/library/tkinter.html

Roden, R., Smith, T., & Sacrey, D. (2015). Geologic pattern recognition from seismic attributes. *Interpretation*, *3*(4), SAE59–SAE83. https://doi.org/10.1190/INT-2015-0037.1

segyio Development Team. (2023). *segyio: Fast and simple SEG-Y file interaction* [Computer software]. https://github.com/equinor/segyio

Sheriff, R. E., & Geldart, L. P. (1995). *Exploration seismology* (2nd ed.). Cambridge University Press.

Vail, P. R., Mitchum, R. M., Jr., & Thompson, S., III. (1977). Seismic stratigraphy and global changes of sea level. In C. E. Payton (Ed.), *Seismic stratigraphy—Applications to hydrocarbon exploration* (pp. 63–81). AAPG Memoir 26.

Virtanen, P., Gommers, R., Oliphant, T. E., Haberland, M., Reddy, T., Cournapeau, D., ... SciPy 1.0 Contributors. (2020). SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods*, *17*, 261–272.

White, R., & Simm, R. (2003). Tutorial: Good practice in well ties. *First Break*, *21*(10), 75–83.

Wrona, T., Pan, I., Gawthorpe, R. L., & Fossen, H. (2018). Seismic facies analysis using machine learning. *Geophysics*, *83*(5), O83–O95. https://doi.org/10.1190/geo2017-0595.1

Wu, X., & Fomel, S. (2018). Automatic fault interpretation with optimal surface voting. *Geophysics*, *83*(5), O67–O82. https://doi.org/10.1190/geo2018-0115.1

Yilmaz, O. (2001). *Seismic data analysis: Processing, inversion, and interpretation of seismic data* (2nd ed.). SEG Investigations in Geophysics No. 10.

Zhao, T., & Mukhopadhyay, P. (2018). A survey on deep learning-based seismic data interpretation. *SEG Technical Program Expanded Abstracts*, *2018*, 1723–1727.
