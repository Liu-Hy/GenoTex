# GenoTEX

Welcome to the official repository of the **Geno**mics Data Au**t**omatic **Ex**ploration Benchmark (GenoTEX), described in our paper [here](https://arxiv.org/abs/2406.15341). GenoTEX supports the evaluation and development of Large Language Model (LLM)-based methods for automating gene expression data analysis, including dataset selection, preprocessing, and statistical analysis.

## Introduction

GenoTEX offers annotated code and results for solving a variety of gene identification questions, organized in a comprehensive analysis pipeline that meets computational genomics standards. These annotations are curated by human bioinformaticians to ensure accuracy and reliability.
You can access the dataset and other resources from this repository to support your research and development in automatic gene data analysis.

<img src="imgs/icon.webp" alt="Data Icon" width="200">

Our work belongs to the general topic of AI4Science, where we show the potential and limitations of LLM-based agents in scientific explorations. 

## Download

- [Input data](https://drive.google.com/drive/folders/1c45TUp5f8nkRbFa7LIIOJv2foN6yBX6c)
- [Preprocessed data](https://drive.google.com/drive/folders/1T-ot3wwVHaAB1NiTesua-Tyqp8CXn7uQ)

## File Structure

### Directories and Files

- **code/**: Contains Jupyter notebooks for the preprocessing of gene expression datasets. Each trait has its own subdirectory with notebooks for specific datasets, named after cohort IDs. The `statistics.py` file provides statistical analysis tools for the preprocessed data.
  
- **preprocessed/**: Includes preprocessed data organized by trait. Each trait subdirectory contains:
  - `cohort_info.json`: Stores results of manual data filtering and metadata such as sample size.
  - `gene_data/`: Subdirectory for preprocessed gene data.
  - `trait_data/`: Subdirectory for preprocessed trait data.

- **output/**: Contains regression results for each trait. Each subdirectory holds results for gene identification problems involving the respective trait, with filenames based on trait-condition pairs.


### Usage

1. **Clone the repository**:
   ```sh
   git clone https://github.com/Liu-Hy/GenoTex.git
   cd GenoTex
2. **Install dependencies**:
   
   Ensure you have the necessary Python packages installed. You can create a virtual environment and install dependencies using:
    ```Python
    python -m venv venv
    source venv/bin/activate 
    pip install -r requirements.txt
    ```

3. **Run code**:
    Navigate to the code/ directory and execute the Jupyter notebooks corresponding to the trait and cohort of interest.

4. **Evaluate performance**:
    Use eval.py to compare the performance of your automated method with the gold standard results provided.

## Contribution
    We welcome contributions to enhance GenoTEX. Please fork the repository, create a new branch for your feature or bug 
    fix, and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

## Citation
If you use GenoTEX in your research, please cite our paper using the following BibTeX entry:
```
@article{liu2024genotex,
  title={GenoTEX: A Benchmark for Evaluating LLM-Based Exploration of Gene Expression Data in Alignment with Bioinformaticians},
  author={Liu, Haoyang and Wang, Haohan},
  journal={arXiv preprint arXiv:2406.15341},
  year={2024}
}
```
## License
    This project is licensed under the Creative Commons (CC) license.