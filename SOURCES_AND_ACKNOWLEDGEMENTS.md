# SOURCES AND ACKNOWLEDGEMENTS
## C.C.R.O.P-PhenoHunt - Cannabis Computational Research & Optimization Platform

---

## Data Sources

### Primary Datasets

#### 1. Leafly Cannabis Strain Profiles
- **Source**: [https://www.leafly.com/strains](https://www.leafly.com/strains)
- **Usage**: Real-world cannabinoid and terpene data scraping
- **Compounds Tracked**: THC, CBD, CBG, CBC, Myrcene, Limonene, Pinene, Caryophyllene, and others
- **License**: Public web data - used for research and educational purposes
- **Acknowledgment**: Chemical composition patterns referenced to simulate realistic cannabinoid-terpene distributions
- **Citation**: Leafly Holdings, Inc. (2025). *Cannabis Strain Database*. Retrieved from https://www.leafly.com

#### 2. Kaggle: Leafly Cannabis Strains Metadata
- **Dataset**: `gthrosa/leafly-cannabis-strains-metadata`
- **Source**: [https://www.kaggle.com/datasets/gthrosa/leafly-cannabis-strains-metadata](https://www.kaggle.com/datasets/gthrosa/leafly-cannabis-strains-metadata)
- **Usage**: Structured training data and validation of compound ranges
- **License**: CC0: Public Domain
- **Acknowledgment**: Used for strain naming conventions and chemical profile validation
- **Citation**: Throsa, G. (2023). *Leafly Cannabis Strains Metadata*. Kaggle.

#### 3. Cannlytics Datasets (Hugging Face)
- **Datasets**: `cannabis_analytes`, `cannabis_results`
- **Source**: [https://huggingface.co/datasets/cannlytics](https://huggingface.co/datasets/cannlytics)
- **Usage**: Cannabis chemical analysis results for model training
- **License**: Check Hugging Face dataset license before use
- **Acknowledgment**: Comprehensive laboratory testing data for cannabinoid and terpene profiles
- **Citation**: Cannlytics (2024). *Cannabis Analytical Results Dataset*. Hugging Face.

#### 4. Kannapedia (Medicinal Genomics)
- **Source**: [https://www.medicinalgenomics.com/kannapedia/](https://www.medicinalgenomics.com/kannapedia/)
- **Usage**: Genomic strain library (requires account/export access)
- **License**: Proprietary - use only with permissions
- **Acknowledgment**: Genomic reference data for strain lineage tracking
- **Note**: Full genomic downloads require authorized access

#### 5. NCBI Genome Assemblies - Cannabis sativa
- **Source**: [https://www.ncbi.nlm.nih.gov/genome/](https://www.ncbi.nlm.nih.gov/genome/)
- **Database**: RefSeq / GenBank
- **Usage**: Public genome assemblies for computational genomics
- **License**: Public domain (U.S. Government work)
- **Acknowledgment**: Reference genome sequences for Cannabis sativa
- **Citation**: National Center for Biotechnology Information (NCBI). *Cannabis sativa Genome Assemblies*. https://www.ncbi.nlm.nih.gov/genome/

---

## Scientific Literature References

### Foundational Research

#### Cannabinoid and Terpene Effects

1. **Russo, E. B. (2011)**
   - *Taming THC: Potential cannabis synergy and phytocannabinoid-terpenoid entourage effects*
   - **Journal**: British Journal of Pharmacology, 163(7), 1344–1364
   - **DOI**: 10.1111/j.1476-5381.2011.01238.x
   - **Impact**: Foundation for entourage effect modeling and compound interaction predictions

2. **Booth, K., & Bohlmann, J. (2019)**
   - *Terpenes in Cannabis sativa – From plant genome to humans*
   - **Journal**: Plant Science, 284, 67–72
   - **DOI**: 10.1016/j.plantsci.2019.03.022
   - **Impact**: Informed terpene biosynthesis and therapeutic associations

3. **Blessing, E. M., Steenkamp, M. M., Manzanares, J., & Marmar, C. R. (2015)**
   - *Cannabidiol as a Potential Treatment for Anxiety Disorders*
   - **Journal**: Neurotherapeutics, 12(4), 825–836
   - **DOI**: 10.1007/s13311-015-0387-1
   - **Impact**: CBD anxiolytic effect coefficients

4. **Ferber, S. G., Namdar, D., Hen-Shoval, D., et al. (2020)**
   - *The "Entourage Effect": Terpenes Coupled with Cannabinoids for the Treatment of Mood Disorders and Anxiety Disorders*
   - **Journal**: Current Neuropharmacology, 18(2), 87–96
   - **DOI**: 10.2174/1570159X17666190903103923
   - **Impact**: Multi-compound synergy modeling

5. **Bonesi, M., Menichini, F., Tundis, R., et al. (2010)**
   - *Acetylcholinesterase and butyrylcholinesterase inhibitory activity of Pinus species essential oils and their constituents*
   - **Journal**: Journal of Enzyme Inhibition and Medicinal Chemistry, 25(5), 622–628
   - **DOI**: 10.3109/14756360903389856
   - **Impact**: α-Pinene neuroprotective mechanisms

#### Chemotype Diversity and Distribution

6. **Smith, C. J., Vergara, D., Keegan, B., & Jikomes, N. (2022)**
   - *The phytochemical diversity of commercial Cannabis in the United States*
   - **Journal**: PLOS ONE, 17(5), e0267498
   - **DOI**: 10.1371/journal.pone.0267498
   - **Impact**: Understanding chemotype distribution and commercial bias; informed data validation ranges

7. **Vergara, D., White, K. H., Keegan, B., & Jikomes, N. (2020)**
   - *Modeling cannabinoids from a large-scale sample of Cannabis sativa chemotypes*
   - **Journal**: PLOS ONE, 15(9), e0236878
   - **DOI**: 10.1371/journal.pone.0236878
   - **Impact**: Statistical modeling approaches for cannabinoid prediction

#### Pharmacology and Therapeutic Mechanisms

8. **Andre, C. M., Hausman, J. F., & Guerriero, G. (2016)**
   - *Cannabis sativa: The Plant of the Thousand and One Molecules*
   - **Journal**: Frontiers in Plant Science, 7, 19
   - **DOI**: 10.3389/fpls.2016.00019
   - **Impact**: Comprehensive molecular profile understanding

9. **ElSohly, M. A., Radwan, M. M., Gul, W., Chandra, S., & Galal, A. (2017)**
   - *Phytochemistry of Cannabis sativa L.*
   - **Journal**: Progress in the Chemistry of Organic Natural Products, 103, 1–36
   - **DOI**: 10.1007/978-3-319-45541-9_1
   - **Impact**: Chemical composition reference standards

10. **Nuutinen, T. (2018)**
    - *Medicinal properties of terpenes found in Cannabis sativa and Humulus lupulus*
    - **Journal**: European Journal of Medicinal Chemistry, 157, 198–228
    - **DOI**: 10.1016/j.ejmech.2018.07.076
    - **Impact**: Individual terpene therapeutic effect coefficients

#### Genetics and Breeding

11. **Vergara, D., Baker, H., Clancy, K., et al. (2016)**
    - *Genetic and Genomic Tools for Cannabis sativa*
    - **Journal**: Critical Reviews in Plant Sciences, 35(5-6), 364–377
    - **DOI**: 10.1080/07352689.2016.1267496
    - **Impact**: Genetic principles for breeding simulation algorithms

12. **Lynch, R. C., Vergara, D., Tittes, S., et al. (2016)**
    - *Genomic and Chemical Diversity in Cannabis*
    - **Journal**: Critical Reviews in Plant Sciences, 35(5-6), 349–363
    - **DOI**: 10.1080/07352689.2016.1265363
    - **Impact**: Chemotype-genotype associations

---

## Software, Libraries, and Frameworks

### Core Scientific Computing
- **Python 3.9+**: Primary programming language
- **NumPy** (>=1.21.0): Numerical computing and array operations
- **Pandas** (>=1.3.0): Data manipulation and analysis
- **SciPy** (>=1.7.0): Scientific computing and statistical functions

### Machine Learning and Deep Learning
- **PyTorch** (>=1.9.0): Variational Autoencoder (VAE) implementation
  - *Citation*: Paszke, A., et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. NeurIPS.
- **Scikit-learn** (>=1.0.0): Ensemble models, cross-validation, preprocessing
  - *Citation*: Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. JMLR.

### Cheminformatics (Planned)
- **RDKit**: 3D molecular structure generation and visualization
- **py3Dmol**: Interactive molecular visualization in Jupyter
- **OpenBabel**: Chemical file format conversion

### Visualization
- **Plotly** (>=5.0.0): Interactive chemical radar graphs and dashboards
- **Matplotlib** (>=3.4.0): Static plots and scientific figures
- **Seaborn** (>=0.11.0): Statistical visualizations
- **PyVista** (Planned): 3D phenotypic visualization

### Data Handling and Processing
- **PyYAML** (>=5.4.0): Configuration management
- **OpenPyXL** (>=3.0.0): Excel file processing
- **FuzzyWuzzy** (>=0.18.0): Approximate string matching for strain recognition
- **python-Levenshtein** (>=0.12.0): Fast string similarity computations

### Web Scraping and Data Acquisition
- **Requests** (>=2.26.0): HTTP library for API calls
- **BeautifulSoup4** (>=4.9.0): HTML parsing for strain data extraction
- **Selenium** (>=3.141.0): Dynamic web scraping

### Document Processing
- **PDFPlumber** (>=0.5.0): Certificate of Analysis (COA) extraction
- **Tabula-py** (>=2.3.0): PDF table extraction

### Interactive Interfaces
- **IPyWidgets** (>=7.6.0): Interactive Jupyter notebook widgets
- **Jupyter Notebook** (>=6.4.0): Development environment

### Testing and Quality Assurance
- **pytest** (>=6.2.0): Testing framework
- **pytest-cov** (>=2.12.0): Code coverage reporting
- **Black** (>=21.0): Code formatting
- **Flake8** (>=3.9.0): Linting and style checking
- **MyPy** (>=0.910): Static type checking

---

## Development and Collaboration

### Original Development
- **Project Concept**: Generative cannabis hybridization using chemical feature fusion
- **Core Architect**: Hosuay (GitHub Project Author)
- **Repository**: [https://github.com/Hosuay/C.C.R.O.P-PhenoHunt](https://github.com/Hosuay/C.C.R.O.P-PhenoHunt)
- **Development Environment**: Google Colab, 2025
- **Version**: Scientific Edition v3.0

### AI-Assisted Development
- **Advisory Schema and Code Design**: Developed collaboratively through AI-assisted prototyping
- **Tools Used**: Claude (Anthropic), for optimizing architecture, feature alignment, and integration
- **Contribution**: Code structure optimization, scientific documentation, testing framework

### Community Inspiration and Acknowledgments

#### Cannabis Breeding and Research Communities
- **Overgrow.com**: Community discussions on breeding techniques
- **THCFarmer**: Phenotype hunting best practices
- **Reddit r/CannabisBreeding**: Open discussions on hybrid strain architecture
- **Grasscity Forums**: Cultivation and breeding knowledge sharing

#### Scientific Open Source Projects
- **QSAR Models for Cannabinoids**: Open-source quantitative structure-activity relationship models
- **Cannabis Genomics Consortium**: Collaborative genomic research efforts

---

## Sacred Geometry and Numerological Framework

### Harmonic Design Principles
This project integrates sacred geometry and numerology principles throughout its architecture:

- **3**: Triadic transformation (Input → Transform → Output)
- **5**: 5-stage preprocessing pipeline
- **7**: 7-layer neural network architectures
- **9**: 9-step post-processing synthesis
- **12**: 12-module repository structure
- **27**: 27 latent partitions for generative strain creation
- **33**: 33 primary chemical and phenotypic features
- **369**: Tesla's divine numbers - ultimate harmonic alignment

### Philosophical Foundation
- **Inspiration**: Nikola Tesla's philosophy: "If you only knew the magnificence of the 3, 6 and 9, then you would have a key to the universe."
- **Application**: Applied to ML training cycles, data partitioning, feature extraction, and system architecture
- **Purpose**: Align computational processes with natural harmonic patterns for optimal performance

---

## Ethical Considerations and Disclaimer

### Research Ethics
- This tool is intended for **educational and research purposes only**
- Does not provide medical advice, legal cultivation guidance, or clinical recommendations
- All predictions are computational hypotheses requiring laboratory validation

### Data Privacy and Compliance
- Use only legally obtained, properly licensed datasets
- For patient outcome data: ensure de-identification and IRB/HIPAA compliance
- Respect intellectual property rights of all data sources
- Provide model cards, versioning, and provenance for all released models

### Regulatory Compliance
- Complies with research regulations in jurisdictions where cannabis research is permitted
- Users are responsible for ensuring compliance with local laws and regulations
- No cultivation or consumption instructions are provided

---

## Citation

If you use C.C.R.O.P-PhenoHunt in your research, please cite:

```bibtex
@software{crop_phenohunt_2025,
  author = {Hosuay and Contributors},
  title = {C.C.R.O.P-PhenoHunt: Cannabis Computational Research & Optimization Platform},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/Hosuay/C.C.R.O.P-PhenoHunt}},
  version = {3.0.0}
}
```

---

## License

This project is released under the **MIT License**.

See [LICENSE](LICENSE) file for full details.

---

## Contact and Contributions

### Project Maintainer
- **GitHub**: [@Hosuay](https://github.com/Hosuay)
- **Repository**: [C.C.R.O.P-PhenoHunt](https://github.com/Hosuay/C.C.R.O.P-PhenoHunt)

### Contributing
We welcome contributions from the scientific community! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Reporting Issues
- **Bug Reports**: [GitHub Issues](https://github.com/Hosuay/C.C.R.O.P-PhenoHunt/issues)
- **Feature Requests**: [GitHub Discussions](https://github.com/Hosuay/C.C.R.O.P-PhenoHunt/discussions)

---

## Acknowledgment of Indigenous Knowledge

We acknowledge that much of the traditional knowledge about Cannabis sativa comes from indigenous peoples and traditional medicine systems spanning thousands of years. This project aims to complement, not replace, traditional knowledge with modern computational methods for scientific research.

---

**Last Updated**: 2025-10-22
**Version**: 3.0.0
**Maintained by**: Hosuay & Contributors
