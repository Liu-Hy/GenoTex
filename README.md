# GenoTEX: An LLM Agent Benchmark for Automated Gene Expression Data Analysis

<div align="center">
  <img src="./imgs/genotex_logo.png" alt="GenoTEX Logo" width="200px"/>
  <br>
  <br>
  <a href="https://creativecommons.org/licenses/by/4.0/">
    <img src="https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg" alt="License: CC BY 4.0">
  </a>
  <a href="https://github.com/topics/ai4science">
    <img src="https://img.shields.io/badge/AI4Science-blue.svg" alt="AI4Science">
  </a>
  <a href="https://github.com/topics/llm-agent">
    <img src="https://img.shields.io/badge/LLM%20Agent-orange.svg" alt="LLM Agent">
  </a>
  <a href="https://github.com/topics/computational-genomics">
    <img src="https://img.shields.io/badge/Computational%20Genomics-green.svg" alt="Computational Genomics">
  </a>
</div>

GenoTEX (**Geno**mics Data Au**t**omatic **Ex**ploration Benchmark) is a benchmark dataset for the automated analysis of gene expression data to identify disease-associated genes while considering the influence of other biological factors. It provides annotated code and results for solving a wide range of gene-trait association (GTA) analysis problems, encompassing dataset selection, preprocessing, and statistical analysis, in a pipeline that follows computational genomics standards. The benchmark includes expert-curated annotations from bioinformaticians to ensure accuracy and reliability.

The below figure illustrates our benchmark curation process. For detailed information, please refer to our [paper on arXiv](https://arxiv.org/abs/2406.15341).

<div align="center">
  <img src="./imgs/benchmark_pipeline.jpg" alt="Benchmark Pipeline" width="90%"/>
</div>

## Table of Contents

- [Dataset Overview](#dataset-overview)
- [Dataset Structure](#dataset-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Citation](#citation)
- [License](#license)

<a id="dataset-overview"></a>
## 📊 Dataset Overview

GenoTEX provides a benchmark for evaluating automated methods for gene expression data analysis, particularly for LLM-based agents. In biomedical research, gene expression analysis is crucial for understanding biological mechanisms and advancing clinical applications such as disease marker identification and personalized medicine. However, these analyses are often repetitive, labor-intensive, and prone to errors, leading to significant time and financial burdens on research teams.

Key features of the benchmark include:

- **1,384 GTA analysis problems**: 132 unconditional problems and 1,252 conditional problems
- **41.5 GB of input data**: 911 datasets with an average of 167 samples per dataset (152,415 total samples)
- **237,907 lines of analysis code**: Carefully curated by bioinformatics experts (average 261 lines per dataset)
- **Three evaluation tasks**: Dataset selection, data preprocessing, and statistical analysis
- **Comprehensive gene features**: Average of 18,530 normalized gene features per dataset
- **Significant gene discoveries**: Significant genes identified per problem

Each problem in the benchmark involves identifying genes associated with a specific trait (e.g., a disease) while optionally considering the influence of some condition (e.g., age, gender, or a co-existing trait). This approach mimics real-life research scenarios where key genes linked to traits often vary based on the diverse physical conditions of patients.

<a id="dataset-structure"></a>
## 🗂️ Dataset Structure

GenoTEX is distributed in two ways:
1. **GitHub repository + Cloud storage**: In this approach, we host the code and documentation in the [GitHub repository](https://github.com/Liu-Hy/GenoTEX), while the data is accessible through separate cloud storage links below. This approach allows for efficient access to the analysis methods and their latest updates, while keeping the large data files separate.
  - Data Available at: [ [Google Drive](https://drive.google.com/drive/folders/1ZQ8AflAecW61SrNclaMby-6x9GLCpJoW) | [Baidu Cloud Disk](https://pan.baidu.com/s/1mKfBRiBNY0GUK6LRLnn7UA?pwd=1234) ] 
  - Total data size: 82.0 GB — Please ensure you have sufficient disk space before downloading.

2. **Complete Datasets on Data Platforms**: We also provide a complete, bundled version (code + data) on [Kaggle](https://www.kaggle.com/datasets/haoyangliu14/genotex-llm-agent-benchmark-for-genomic-analysis) and [Hugging Face Hub](https://huggingface.co/datasets/Liu-Hy/GenoTEX).
These versions are convenient for users who prefer a single download and want to leverage the functionalities of these platforms. 

### The Data Part

The data is organized into three main directories:

```
./
├── input/               # Raw input data from public databases
│   ├── GEO/             # Data from Gene Expression Omnibus
│   │   ├── Trait_1/
│   │   │   ├── GSE12345/
│   │   │   │   ├── GSE12345_family.soft.gz
│   │   │   │   └── GSE12345_series_matrix.txt.gz
│   │   │   └── ...
│   │   └── ...
│   └── TCGA/            # Data from The Cancer Genome Atlas
│       ├── TCGA_Cancer_Type_1/
│       │   ├── TCGA.XXXX.sampleMap_XXXX_clinicalMatrix
│       │   └── TCGA.XXXX.sampleMap_HiSeqV2_PANCAN.gz
│       └── ...
│
├── output/              # Analysis output data
│   ├── preprocess/      # Preprocessed datasets
│   │   ├── Trait_1/
│   │   │   ├── clinical_data/
│   │   │   │   ├── GSE12345.csv
│   │   │   │   ├── Xena.csv
│   │   │   │   └── ...
│   │   │   ├── gene_data/
│   │   │   │   ├── GSE12345.csv
│   │   │   │   ├── Xena.csv
│   │   │   │   └── ...
│   │   │   ├── cohort_info.json
│   │   │   ├── GSE12345.csv
│   │   │   ├── Xena.csv
│   │   │   └── ...
│   │   └── ...
│   └── regress/         # Regression analysis results
│       ├── Trait_1/
│       │   ├── significant_genes_condition_None.json
│       │   ├── significant_genes_condition_Condition_1.json
│       │   └── ...
│       └── ...
│
└── metadata/            # Problem specifications and domain knowledge
    ├── task_info.json   # GTA analysis problems; domain knowledge about gene-trait associations
    └── gene_synonym.json # Gene symbol mapping
```

### Notes on Data Organization:

**1. Trait Name Normalization**: To make path operations safer, trait names are normalized by removing apostrophes (') and replacing spaces with underscores (_). For example, "Crohn's Disease" becomes "Crohns_Disease", and "Large B-cell Lymphoma" becomes "Large_B-cell_Lymphoma". This consistent formatting helps prevent path-related errors when working with the dataset.

**2. Input Data Structure**:
   
   - **GEO folder**: Organized by our predefined trait names. Each trait directory holds 1-11 cohort datasets related to this trait, which were programmatically searched under specific criteria and downloaded from [the GEO database](https://www.ncbi.nlm.nih.gov/geo/) using [Entrez Utilities](https://www.ncbi.nlm.nih.gov/books/NBK25501/). Each cohort dataset is stored as a folder named after the GEO Series (GSE) number, such as 'GSE98578'. Each cohort folder contains `.gz` files, typically one SOFT file and one matrix file, though occasionally there are multiple SOFT files or matrix files.

   - **TCGA folder**: Directly downloaded from [the TCGA Hub](https://xenabrowser.net/datapages/?host=https%3A%2F%2Ftcga.xenahubs.net&removeHub=https%3A%2F%2Fxena.treehouse.gi.ucsc.edu%3A443) of [the UCSC Xena platform](https://xena.ucsc.edu/), organized by different types of cancer. We preserve the original folder names from the website, without strictly matching our predefined trait names. Each trait (cancer) folder contains a clinicalMatrix file storing sample clinical features, and a PANCAN.gz file storing gene expression data.

<a id="preprocessing-results-structure"></a>
**3. Preprocessing Results Structure**:
   
   The 'output/preprocess/' folder is organized by predefined trait names. For each trait, it holds the clinical data, gene expression data, and the final linked data that are successfully preprocessed for each cohort into 3 separate CSV files. These CSV files are saved in '{trait_name}/clinical_data/', '{trait_name}/gene_data/', and '{trait_name}/' respectively, with the same filename '{cohort_ID}.csv'. 
   For GEO cohorts, the cohort ID is the Series number (GSE); for TCGA, since there is at most one TCGA cohort related to each trait, we directly use 'Xena' as the cohort ID.

   Additionally, each trait subfolder contains a JSON file, which stores metadata about dataset filtering and preprocessing outcomes of all related cohorts.

#### Example Data Formats:
   
   **a. Gene Expression Data** (stored in `{trait_name}/gene_data/{cohort_ID}.csv`):
   
   This file contains gene expression values with gene symbols as rows and sample IDs as columns:
   
   ```
            GSM7920782  GSM7920783  GSM7920784  ...  GSM7920825
   A2M          13.210      13.238      14.729  ...       7.359
   ACVR1C        5.128       5.337       5.611  ...       8.151
   ADAM12        9.807      12.374       9.953  ...       9.266
   ...             ...         ...         ...  ...         ...
   ZEB2          9.534      10.488      10.553  ...       8.151
   ```
   
   **b. Clinical Data** (stored in `{trait_name}/clinical_data/{cohort_ID}.csv`):
   
   This file contains clinical information with trait names as rows and sample IDs as columns:
   
   ```
                 GSM7920782  GSM7920783  ...  GSM7920825
   Breast_Cancer        1.0         1.0  ...         0.0
   Age                 49.0        44.0  ...        74.0
   Gender               0.0         0.0  ...         1.0
   ```
   
   **c. Linked Dataset** (stored in `{trait_name}/{cohort_ID}.csv`):
   
   This file combines clinical and gene expression data with samples as rows and all features (clinical and gene) as columns:
   
   ```
              Breast_Cancer    Age  Gender     A2M    ACVR1C    ADAM12  ...     ZEB2
   GSM7920782           1.0   49.0     0.0  13.210     5.128     9.807   ...   9.534
   GSM7920783           1.0   44.0     0.0  13.238     5.337    12.374   ...  10.488
   ...                   ...   ...     ...     ...       ...       ...   ...     ...
   GSM7920825           0.0   74.0     1.0   7.359     8.151     9.266   ...   8.151
   ```
   
   **d. Cohort Information** (stored in `{trait_name}/cohort_info.json`):
   
This file contains outputs of dataset filtering (initial filtering and quality verification), and metadata about the preprocessing outcomes of each cohort related to the trait:
   
   ```json
   {
   "GSE207847": 
     {"is_usable": true, "is_gene_available": true, "is_trait_available": true, "is_available": true, "is_biased": false, "has_age": false, "has_gender": false, "sample_size": 60},
   "GSE153316":
     {...},
   ...
   }
   ```
   
   Each cohort entry contains the following fields:
   - `is_gene_available` and `is_trait_available`: Indicate whether the dataset has the relevant gene expression and trait information, respectively. `is_available` is true if both are available.
   - `is_biased`: Indicates whether the trait distribution is severely biased. For example, if a dataset about breast cancer treatment only contains positive samples, it would be considered biased and not usable for trait-gene association studies.
   - `is_usable`: True if `is_available` is true and `is_biased` is false.
   - `has_age` and `has_gender`: Indicate whether the dataset contains the samples' age and gender information, respectively.
   - `sample_size`: Records the number of samples in the final linked dataset, after discarding samples with too many missing features.

<a id="regression-results-structure"></a>
**4. Regression Results Structure**:
   
   The 'output/regress/' folder is also organized by predefined trait names. It holds the regression analysis outputs for all GTA analysis problems in our benchmark that involve the same trait. These problems are uniquely identified by a trait-condition pair.
   
   The analysis output for each problem is stored in a file named "significant_genes_condition_{condition name}.json", where the condition name is either a predefined trait name, or 'Age', 'Gender', or 'None'. A 'None' condition represents an unconditional problem—"What are the significant genes related to this trait?"—without considering the influence of any conditions.
   
   Each JSON file contains:
   
   ```json
   {
       "significant_genes": {
           "Variable": ["TRIB1", "MTMR14", "ANKFY1", ...],
           "Coefficient": [-3.6007, 2.7751, -2.5880, ...],
           "Absolute Coefficient": [3.6007, 2.7751, 2.5880, ...]
       },
       "cv_performance": {...}
   }
   ```
   
   - `significant_genes`: A dictionary with three keys:
     - `Variable`: List of gene symbols identified as significant
     - `Coefficient`: The corresponding coefficients in the trained regression model
     - `Absolute Coefficient`: The absolute values of these coefficients
   
   The gene symbols are ranked by importance (absolute coefficient in Lasso models). The `cv_performance` part is used mainly for model validation and diagnostics, not part of our benchmark evaluation.

**5. Metadata Structure**:
   
   - `task_info.json`: Contains full specifications for the GTA analysis problems in our benchmark, and domain knowledge about gene-trait associations. For each trait, it includes:
   
      ```json
      {
          "Acute_Myeloid_Leukemia": 
          {
              "related_genes": ["DNMT3A", "FLT3", "CEBPA", "TET2", "KIT", ... ],
              "conditions": ["Age", "Gender", "Eczema", ... ]    
          },
          "Adrenocortical_Cancer": {
              ...
          }
      }
      ```
   
     - `related_genes`: A list of genes known to be associated with the trait, sourced from [the Open Targets Platform](https://platform.opentargets.org/downloads)
     - `conditions`: The list of conditions paired with the trait to form the GTA analysis problems in our benchmark
   
   - `gene_synonym.json`: Stores the mapping from common acronyms of human gene symbols to their standard symbols, sourced from [the NCBI Gene FTP Site](https://ftp.ncbi.nlm.nih.gov/gene/DATA/). This is useful for normalizing gene symbols during preprocessing to prevent inaccuracies arising from different gene naming conventions. Standard symbols are mapped to themselves.
   
      ```json
      {
          "Acronym_1": "Standard_Symbol",
          "Acronym_2": "Standard_Symbol",
          "Standard_Symbol": "Standard_Symbol",
          ...
      }
      ```
   

### The Code Part

```
./
├── code/                # Analysis code
│   ├── Trait_1/
│   │   ├── GSE12345.ipynb
│   │   ├── Xena.ipynb
│   │   └── ...
│   ├── ...
│   └── regress.py       # Regression analysis implementation
│
├── tools/               # Function tools for gene expression data analysis
├── utils/               # Helper functions for experimentation and evaluation
├── download/            # Scripts for downloading datasets
├── datasheet.md         # Datasheets for Datasets documentation
├── metadata.json        # Croissant metadata in JSON-LD format
└── requirements.txt     # Package dependencies
```

The code part of the benchmark includes:

- **code/**: Contains our code for gene expression data analysis. The main part is the code for preprocessing each cohort dataset, organized by predefined trait names. We provide the code as Jupyter Notebook files with the name '{cohort_ID}.ipynb', showing the output of each step to facilitate interactive analysis. `regress.py` implements our regression analysis method in fixed logic, for solving the GTA analysis problems in our benchmark.

- **tools/**: Contains the function tools that are accessible to both human bioinformaticians and LLM agents for gene expression data analysis.

- **utils/**: Contains the helper functions used for this project outside of the data analysis tasks, e.g., experiment logging, evaluation metrics, etc.

- **download/**: Contains the scripts for programmatically searching and downloading input gene expression datasets, and acquiring domain knowledge files from public repositories. It also includes the script for selecting important trait-condition pairs to form our GTA analysis problems.

- **Documentation files**: `datasheet.md` provides the [Datasheets for Datasets](https://arxiv.org/abs/1803.09010) documentation of our benchmark, and `metadata.json` provides [the Croissant metadata](https://github.com/mlcommons/croissant) in [JSON-LD](https://json-ld.org/) format.

<a id="installation"></a>
## 📥 Installation
**1. Download the dataset**

- **For the GitHub version**

    (1) Clone this repository:

      ```bash
      git clone https://github.com/Liu-Hy/GenoTEX.git
      cd GenoTEX
      ```
    (2) Download the data folders ('metadata', 'input', 'output') from the provided cloud storage links, and place them in the root directory of this repository.
- **For the bundled version**

  Download the dataset folder containing code and data directly from the data platform.

**2. (For Kaggle users only) Recompress files** 

Kaggle automatically unzips all `.gz` files, but our code requires certain files to remain compressed. Run the provided script to recompress these files (this will also save significant disk space):
   ```bash
   python recompress_files.py
   ```

**3. Create and activate a conda environment**
   ```bash
   conda create -n genotex python=3.10
   conda activate genotex
   pip install -r requirements.txt
   ```

<a id="usage"></a>
## 💻 Usage

### Exploring the Benchmark

The GenoTEX benchmark code is organized into two complementary components. First, you'll find Jupyter notebooks in the `./code/{trait_name}/` directories that handle dataset-specific preprocessing. These notebooks prepare individual datasets by cleaning, standardizing, and integrating the data, but they don't perform the actual association studies.

The statistical analysis that identifies genes associated with traits is centralized in the `./code/regress.py` script. This script automatically selects the optimal cohort(s) from all preprocessed datasets for each problem, applies appropriate statistical models, and performs hyperparameter tuning to identify significant genes.

This design separates data preparation from statistical modeling, enabling consistent methodology across all analyses while maximizing statistical power through automatic cohort selection. To run the complete pipeline, first execute the preprocessing notebooks for individual datasets, then run the regress.py script to perform association studies across all problems.

### Evaluating Your Own Methods

After developing your automated method for gene expression data analysis, you can evaluate it against our benchmark:

1. Ensure your method produces output following the same format as our benchmark. Your output should be organized in the same structure as our `./output` directory, with:

   - Preprocessed datasets and JSON files in the same folder structure and file format as described in [Preprocessing Results Structure](#preprocessing-results-structure)
   - Regression results with gene lists ranked by importance, in the same JSON format as described in [Regression Results Structure](#regression-results-structure)

2. Run the evaluation script:
   ```bash
   python eval.py -p {prediction_directory} -r {reference_directory} -t selection preprocessing analysis -s gene clinical linked
   ```

   Where:
   - `-p`, `--pred-dir`: Path to the directory containing your prediction results, required.
   - `-r`, `--ref-dir`: Path to the directory containing reference benchmark results (default: "./output")
   - `-t`, `--tasks`: Specific tasks to evaluate: "selection" (dataset filtering and selection), "preprocessing" (data preprocessing), "analysis" (statistical analysis) - omit to evaluate all tasks
   - `-s`, `--preprocess-subtasks`: Specific preprocessing aspects to evaluate: "gene" (expression data), "clinical" (trait data), "linked" (final linked data) - omit to evaluate all subtasks

The evaluation produces metrics for each task:

- **Dataset selection and filtering**: Evaluates the ability to identify useful datasets and select the optimal ones for analysis
- **Data preprocessing**: Measures how well the method processes and integrates data from different sources
- **Statistical analysis**: Assesses the accuracy of identifying significant genes related to traits

The script will report errors arising from non-conformance in format, but you will need to correct any formatting issues to ensure accurate evaluation.

<a id="citation"></a>
## 📝 Citation

If you use GenoTEX in your research, please cite our paper:

```bibtex
@misc{liu2025genotex,
      title={GenoTEX: A Benchmark for Automated Gene Expression Data Analysis in Alignment with Bioinformaticians}, 
      author={Haoyang Liu and Shuyu Chen and Ye Zhang and Haohan Wang},
      year={2025},
      eprint={2406.15341},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2406.15341}, 
}
```

<a id="license"></a>
## ⚖️ License

The GenoTEX dataset is released under a Creative Commons Attribution 4.0 International (CC BY 4.0) license, which allows for broad usage while protecting the rights of the creators. The authors bear full responsibility for ensuring that the dataset adheres to this license and for any potential violations of rights. For the full license text, please see the [LICENSE.md](./LICENSE.md) file in this repository.
