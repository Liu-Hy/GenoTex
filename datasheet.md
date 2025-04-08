# GenoTEX Datasheet

## Motivation

**For what purpose was the dataset created?**

The GenoTEX dataset was created to support the evaluation and development of automated methods for gene expression data analysis, particularly for LLM-based agents. In biomedical research, gene expression analysis is crucial for understanding biological mechanisms and advancing clinical applications such as disease marker identification and personalized medicine. However, these analyses are often repetitive, labor-intensive, and prone to errors, leading to significant time and financial burdens on research teams (estimated at around $848.3 million annually, with costs expected to increase at a CAGR of 12% to 16% by 2030). GenoTEX aims to facilitate the advancement of AI methods capable of automating these complex tasks, addressing the need for more efficient and cost-effective data analysis solutions in genetics research.

**Who created the dataset (e.g., which team, research group) and on behalf of which entity (e.g., company, institution, organization)?**

The dataset was created by a team of researchers led by Haohan Wang from the UIUC DREAM Lab, with core members including Haoyang Liu, Shuyu Chen, and Ye Zhang. The project was conducted as part of their research in AI4Science, specifically focusing on AI-driven methods for biomedical data analysis.

**Who funded the creation of the dataset?** 

This research was supported by the National AI Research Resource (NAIRR) under grant number 240283. An initial version of this work was supported by the Accelerate Foundation Models Research (AFMR) initiative from Microsoft Research.

## Composition

**What do the instances that comprise the dataset represent?**

The dataset represents 1,384 gene-trait association (GTA) analysis problems, each uniquely identified by a (trait, condition) pair. Each problem represents a scientific inquiry to identify significant genes associated with a specific trait (e.g., a disease) while accounting for the influence of a condition (e.g., age, gender, or another trait). The condition is either another trait, or 'Age', 'Gender', or 'None' for unconditional problems.

**How many instances are there in total (of each type, if appropriate)?**

The dataset includes:
- 1,384 GTA analysis problems (132 unconditional problems and 1,252 conditional problems)
- 911 input datasets from GEO and TCGA related to 132 traits
- 41.5 GB of input data with 152,415 total samples (average of 167 samples per dataset)
- 237,907 lines of analysis code (average 261 lines per dataset)

**Does the dataset contain all possible instances or is it a sample from a larger set?**

The dataset is a sample from a larger set. Our [sampling strategy](#sampling-strategy) is answered in a following question.

**What data does each instance consist of?**

For each problem, the dataset contains:

1. Input data: Raw gene expression datasets from public databases (GEO and TCGA) associated with the relevant trait (and condition, if applicable). The raw data includes gene expression measurements from multiple samples, along with clinical information about these samples.
2. Analysis code: Annotated code for data preprocessing and statistical analysis
3. Output data: Preprocessed datasets and the significant genes identified from statistical analyses

Note that each GTA analysis problem reuses datasets related to its trait and condition. For example, if problem_1 is (trait_A, trait_B) and problem_2 is (trait_A, trait_C), then both problems use the preprocessed datasets for trait_A. 

**Is there a label or target associated with each instance?**

Yes, each GTA analysis problem has associated "reference answer" in the form of significant genes identified by expert bioinformaticians following a standardized analysis pipeline. These are stored in JSON files with gene symbols, their coefficients, and absolute coefficients in the trained regression model.

**Is any information missing from individual instances?**

Some datasets may have missing information, such as age or gender data for certain samples. The `cohort_info.json` files document which additional clinical features (e.g., age, gender) are available for each dataset.

**Are relationships between individual instances made explicit?**

Yes, relationships between problems, conditions, and traits are explicitly documented in the metadata. The `task_info.json` file maps each trait to its related genes and conditions, making the relationship structure clear.

**Are there recommended data splits?**

No, GenoTEX does not specify training/validation/testing splits. The benchmark evaluates automated methods for gene expression data analysis, which typically employ agents that leverage the reasoning and programming capabilities of foundation models, either with or without additional fine-tuning. Since these methods do not rely on supervised learning from the benchmark itself, traditional data splits are unnecessary.

**Are there any errors, sources of noise, or redundancies in the dataset?**

The dataset acknowledges inherent ambiguity in gene selection due to specific choices made during cohort-specific feature encoding, where multiple reasonable approaches often exist. However, the high Inter-Annotator Agreement (IAA) with an F₁ score of 94.73% for dataset filtering and AUROC score of 0.89 demonstrates high consistency among annotators, validating the reliability of the benchmark.

**Is the dataset self-contained, or does it link to or otherwise rely on external resources?**

The dataset is mostly self-contained, with all necessary components included. The raw data is sourced from public repositories (GEO and TCGA), and gene-trait associations and gene synonym mappings are included in the metadata directory. The original sources (GEO, TCGA, Open Targets Platform, NCBI Genes database) are still maintained by their respective organizations and should remain available, though the specific versions used in GenoTEX are captured in the dataset to ensure reproducibility.

**Does the dataset contain data that might be considered confidential?**

No, the dataset does not contain confidential data. All source data was obtained from public repositories, and the authors have ensured that no personally identifiable information is included.

**Does the dataset contain data that, if viewed directly, might be offensive, insulting, threatening, or might otherwise cause anxiety?**

No, the dataset contains gene expression data and clinical information from anonymized samples for scientific research purposes and does not contain offensive or anxiety-inducing content.

**Does the dataset identify any subpopulations?**

The dataset includes demographic information such as age and gender where available in the source data. The `cohort_info.json` files indicate which datasets contain age and gender information for their samples.

**Is it possible to identify individuals from the dataset?**

No, the dataset does not allow for the identification of individuals. Throughout the curation process, the authors carefully examined each dataset to ensure the absence of personally identifiable information and compliance with all relevant standards.

**Does the dataset contain data that might be considered sensitive in any way?**

The dataset contains health-related data (gene expression and disease status), but these are anonymized and aggregated from public sources. The authors have taken care to ensure compliance with ethical standards for working with such data.

## Collection Process

**How was the data associated with each instance acquired?**

The input data was obtained from public gene expression databases: The Gene Expression Omnibus (GEO) and The Cancer Genome Atlas (TCGA). Domain knowledge was acquired from the Open Targets Platform for gene-trait associations and the NCBI Genes database for gene synonym mapping.

The analysis code and results were directly created by a team of bioinformaticians following standardized guidelines developed by the research team.

**What mechanisms or procedures were used to collect the data?**

The input datasets were programmatically searched and downloaded from the GEO database using Entrez Utilities and from the TCGA Hub of the UCSC Xena platform. The scripts used for this process are included in the './download/' directory of the repository.

For the analysis part, a team of 4 researchers designed the problem list and developed example code for solving GTA analysis problems. They extracted common patterns from these examples to develop guidelines for the entire benchmark. Then, a team of 9 bioinformaticians was assembled and trained to analyze the complete set of problems following these guidelines. They submitted their analysis code and results weekly over a period of 20 weeks.

<a id="sampling-strategy"></a>**If the dataset is a sample from a larger set, what was the sampling strategy?**

The sampling strategy involved two aspects:

1. **Problem selection**: The selection of trait-condition pairs involved both domain expertise and data-driven approaches. The researchers applied manually designed rules to determine which pairs to include or exclude based on trait categories. For undecided pairs, they measured trait-condition association by calculating Jaccard similarity between gene sets related to each trait and condition, using data from the Open Targets Platform. They selected pairs with Jaccard similarity exceeding 0.1, as these likely share underlying genetic mechanisms, offering valuable insights into complex trait-condition interactions.

2. **Dataset collection**: For each trait, the researchers selectively downloaded gene expression datasets:
   - **GEO data**: For each trait, only the top 10 cohort datasets that satisfied specific criteria were downloaded, using GEO's default ranking that considers recency and quality. The search was limited to GEO's manually curated subset (GEO DataSets) with the following criteria:
     - Sample size: 30-10,000
     - Organism: Human
     - Publication year: 2010-2025
     - Data types: "expression profiling by array", "expression profiling by high throughput sequencing", "genome variation profiling by high throughput sequencing", "genome variation profiling by snp array", "snp genotyping by snp array", "third party reanalysis"
     - Technical requirements: Must have both a matrix file (>0.1MB, <100MB) and a family file (<100MB)
   - **TCGA data**: All available data from the UCSC Xena Data Hubs were included due to their reasonable size and high quality.

**Who was involved in the data collection process and how were they compensated?**

The data collection and analysis involved a core team of 4 researchers who designed the problem list and guidelines, and a team of 9 bioinformaticians who conducted the analyses. Information about compensation is not provided in the available materials.

**Over what timeframe was the data collected?**

The analysis code and results were developed and submitted by bioinformaticians over two phases totaling 20 weeks: the main development phase from February to May 2024, and a supplementary phase in January 2025 to incorporate more recent data.

**Were any ethical review processes conducted?**

The authors mention engaging in extensive discussions and consultations to address ethical considerations and legal requirements throughout the curation process. They carefully examined each dataset to ensure the absence of personally identifiable information and compliance with all relevant standards.

## Preprocessing/cleaning/labeling

**Was any preprocessing/cleaning/labeling of the data done?**

Yes, extensive preprocessing was performed on the raw gene expression data according to a standardized pipeline. The preprocessing steps included:

1. **Dataset filtering and selection**: Filtering out irrelevant datasets and selecting the best dataset for each GTA analysis problem based on relevance, quality, and sample size.

2. **Gene expression data preprocessing**: For microarray data, starting with raw datasets identified by probe IDs and mapping these to gene symbols using platform-specific gene annotation data. For RNA-seq data, handling sequence reads that require alignment to a reference genome. Normalizing and deduplicating gene symbols by querying a local gene database to prevent inaccuracies arising from different gene naming conventions.

3. **Trait data extraction**: Identifying attributes containing variable information of interest, designing conversion rules, and writing functions to encode attributes into binary or numerical variables. This often required inferring information based on an understanding of the data measurement and collection process described in the metadata, combined with necessary domain knowledge.

4. **Data linking**: Linking the preprocessed gene data with the extracted trait data based on sample IDs to create a data table containing both genetic and clinical features for the same samples.

The preprocessing also involved common operations such as missing value imputation and column matching.

**Was the "raw" data saved in addition to the preprocessed/cleaned/labeled data?**

Yes, both the raw data (in the 'input' directory) and the preprocessed data (in the 'output/preprocess' directory) are included in the dataset.

**Is the software used to preprocess/clean/label the data available?**

Yes, the code used for preprocessing is available in the 'code/' directory of the repository, organized by predefined trait names. Each file (e.g., 'GSE12345.ipynb') contains the code for preprocessing a specific cohort dataset. The 'tools/' directory contains function tools used for gene expression data analysis.

## Uses

**Has the dataset been used for any tasks already?**

Yes, GenoTEX has been used to evaluate GenoAgent, a team of LLM-based agents proposed by the authors as a baseline method for automating gene expression data analysis. The evaluation assessed performance on three tasks: dataset selection, data preprocessing, and statistical analysis.

**Is there a repository that links to any or all papers or systems that use the dataset?**

The GitHub repository (https://github.com/Liu-Hy/GenoTEX) serves as the primary hub for the dataset and includes references to the associated paper.

**What (other) tasks could the dataset be used for?**

GenoTEX could be used for:
1. Developing and evaluating automated methods for gene expression data analysis
2. Training machine learning models to identify disease-associated genes
3. Studying the influence of conditions on gene-trait relationships
4. Benchmarking different approaches to dataset selection, preprocessing, and statistical analysis in genomics
5. Teaching and educational purposes in bioinformatics and computational genomics

**Is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might impact future uses?**

GenoTEX includes expert-curated annotations following best practices in the bioinformatics community. However, the set of selected genes is sensitive to specific choices made during cohort-specific feature encoding, where multiple reasonable approaches often exist. This inherent ambiguity in gene selection should be considered when evaluating methods against this benchmark.

**Are there tasks for which the dataset should not be used?**

GenoTEX should NOT be used for:
1. Making clinical decisions without additional validation through biological experiments or clinical trials
2. Claiming definitive "ground truth" about gene-disease relationships, as these analyses provide valuable insights but must ultimately be combined with interventional biological experiments or clinical trials to confirm the significance of identified genes
3. Developing methods that ignore the inherent complexity and ambiguity in gene expression analysis

## Distribution

**Will the dataset be distributed to third parties outside of the entity on behalf of which the dataset was created?**

Yes, GenoTEX is publicly available for research purposes.

**How will the dataset will be distributed? Does the dataset have a digital object identifier (DOI)?**

GenoTEX is distributed in two main ways:

1.  **GitHub Repository + Cloud Storage**: The [GitHub repository](https://github.com/Liu-Hy/GenoTEX) hosts the code and documentation, with data accessible via cloud storage links (Google Drive/Baidu Cloud Disk). This is good for accessing the latest code updates.
2.  **Complete Bundled Datasets**: Available on [Kaggle](https://www.kaggle.com/datasets/haoyangliu14/genotex-llm-agent-benchmark-for-genomic-analysis) and [Hugging Face Hub](https://huggingface.co/datasets/Liu-Hy/GenoTEX), containing both code and data for convenience.

The dataset's DOI is: [https://doi.org/10.34740/kaggle/dsv/11309048](https://doi.org/10.34740/kaggle/dsv/11309048)

**When will the dataset be distributed?**

GenoTEX has already been distributed, as indicated by the availability of the GitHub repository and the publication of the associated paper on arXiv.

**Will the dataset be distributed under a copyright or other intellectual property (IP) license, and/or under applicable terms of use (ToU)?**

Yes, GenoTEX is released under a Creative Commons Attribution 4.0 International (CC BY 4.0) license, which allows for broad usage while protecting the rights of the creators.

**Have any third parties imposed IP-based or other restrictions on the data associated with the instances?**

The original data sources (GEO, TCGA, Open Targets Platform, NCBI Genes database) are public resources with their own terms of use, but they generally allow for research use with proper attribution.

**Do any export controls or other regulatory restrictions apply to the dataset or to individual instances?**

No, there are no export controls or other regulatory restrictions that apply to GenoTEX. The dataset consists of publicly available gene expression data and analysis code, which are not subject to special regulatory controls.

## Maintenance

**Who will be supporting/hosting/maintaining the dataset?**

GenoTEX is hosted on GitHub and maintained by Haoyang Liu (the first author) and other researchers from the UIUC DREAM Lab.

**How can the owner/curator/manager of the dataset be contacted?**

Users are welcome to discuss issues and/or make pull requests on the GitHub repository. For specific inquiries or collaborations, users can contact haoyang.liu.ted@foxmail.com.

**Is there an erratum?**

We do not provide an explicit erratum. However, we will address any identified or reported issues in the dataset and make timely updates. We will provide changelogs to document the updates between stable releases.

**Will the dataset be updated?**

Yes, we will continue to update GenoTEX based on feedback from the community and our subsequent research findings related to the dataset.

**If the dataset relates to people, are there applicable limits on the retention of the data associated with the instances?**

GenoTEX contains anonymized data from public repositories with no personally identifiable information. We have not set specific retention limits as this data is already publicly available.

**If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so?**

Contributions can be made through the GitHub repository using standard mechanisms such as pull requests. 