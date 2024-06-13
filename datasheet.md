# GenoTEX Datasheet

## Motivation

**For what purpose was the dataset created?**
The GenoTEX dataset was created to support the evaluation and development of AI-driven methods for the automatic analysis of gene expression data. It aims to facilitate the advancement of machine learning models capable of automating the complex task of gene expression analysis.

**Who created the dataset and who funded it?**
The dataset was created by a team of researchers led by Haoyang and his PhD advisor, Haohan. The project was supported by their research institution.

**What are the applications of this dataset?**
The primary application of GenoTEX is to benchmark gene expression analysis algorithms. It is intended for use in research in bioinformatics, computational genomics, and machine learning.

## Composition

**What do the instances that comprise the dataset represent?**
Each instance in the dataset represents a gene expression measurement from a biological sample, along with associated clinical traits.

**How many instances are there?**
The dataset includes 795 gene expression datasets with a total of 132,673 samples.

**What data does each instance consist of?**
Each instance consists of:
- Gene expression levels
- Clinical traits and conditions
- Metadata describing the dataset

## Collection Process

**How was the data collected?**
Data were collected from publicly available gene expression databases, including The Cancer Genome Atlas (TCGA) via the Xena platform and the Gene Expression Omnibus (GEO).

**What mechanisms or procedures were used to collect the data?**
The data were downloaded using APIs provided by the respective databases. Relevant datasets were selected based on predefined criteria related to the traits and conditions of interest.

## Preprocessing/Cleaning/Labeling

**What preprocessing or cleaning was done?**
Preprocessing steps included:
- Mapping initial identifiers to gene symbols using platform-specific gene annotation data.
- Normalizing and deduplicating gene symbols by querying gene databases.
- Encoding clinical traits into binary, ordinal, or categorical variables.

**Was any labeling applied to the data?**
Yes, clinical traits and conditions were labeled according to predefined rules and expert domain knowledge.

## Uses

**What are the intended uses of the dataset?**
The dataset is intended for benchmarking gene expression analysis algorithms and facilitating research in bioinformatics and machine learning.

**Who are the intended users?**
The primary users are researchers in the fields of bioinformatics, computational genomics, and machine learning.

## Distribution

**How is the dataset distributed?**
The dataset is publicly available on GitHub.

**Are there any licenses or terms of use?**
The dataset is released under a Creative Commons (CC) license. Detailed information about the license can be found [here](https://example.com/license).

## Maintenance

**Who is responsible for maintaining the dataset?**
The research team led by Haoyang and his PhD advisor, Haohan, is responsible for maintaining the dataset.

**How will updates be communicated?**
Updates will be communicated through the GitHub repository where the dataset is hosted.

## Ethical Considerations

**What ethical considerations are associated with the dataset?**
The dataset has been curated to ensure that it does not include any personally identifiable information. Ethical implications and guidelines for responsible use are provided.

**Legal Compliance**
The dataset complies with all relevant legal requirements, and the authors bear full responsibility for ensuring this compliance.

## Data Format and Metadata

**What is the format of the data?**
The data are provided in open and widely used formats such as CSV and JSON.

**Is there any additional metadata?**
Yes, the metadata is documented using the Croissant Metadata Record. The metadata can be accessed [here](https://github.com/Liu-Hy/GenoTEX/blob/main/metadata.json).

## Persistent Identifiers

**Is there a persistent identifier for the dataset?**
A DOI has been minted for GenoTEX to ensure persistent access and citation.

## Reproducibility

**How is reproducibility supported?**
All necessary datasets, code, and evaluation procedures are included in the documentation to ensure that others can replicate the results of our analyses.

## Contact Information

**Who can be contacted for further questions?**
For further questions, please contact Haoyang at hl57@illinois.edu

