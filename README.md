# Ambiguity Scoring of Candidate Terms Based on Their Contextual Embeddings

## Project Overview

This repository contains the code and data for the Master's thesis titled "Ambiguity scoring of candidate terms based upon their contextual embeddings" by Dimitrios Papadopoulos. The thesis was conducted during a Master internship with Elsevier B.V. and focuses on leveraging pre-trained transformer models to distinguish between ambiguous and unambiguous terms within scientific taxonomies. This work is part of the MSc Information Studies program at the University of Amsterdam (UvA).

## Abstract

Term ambiguity poses a significant challenge in scientific literature and taxonomy production, with some terms carrying multiple meanings based on their context. This study addresses this gap by leveraging the pre-trained transformer models, SciBERT and T5, to distinguish between ambiguous and unambiguous terms within three scientific taxonomies. When fine-tuned for the specific task or paired with traditional machine learning classifiers, these transformer models effectively distinguished between ambiguous and unambiguous terms, achieving high F1 scores, reaching up to 78.47% on the Omniscience dataset and commendable F1 scores up to 68.51% for Emtree/RMC and 55.36% for MeSH. The findings demonstrate the potential of transformer models in handling term ambiguity in scientific taxonomies and offer insights that could significantly enhance the construction and refinement of scientific taxonomies and improve information retrieval in scientific research.

## Table of Contents

- [Installation](#installation)
- [Data](#data)
- [Usage](#usage)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Installation

To set up the project on your local machine, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/ambiguity-scoring.git
   cd ambiguity-scoring

2. **Create and activate a Conda environment:**  
   ```bash  
   conda env create -f environment.yml  
   conda activate myenv   

## Data  

The datasets used in this study include Emtree/RMC and Omniscience, which are internal taxonomies and cannot be published. For reproducibility, you can use the MeSH dataset from the paper by Tsatsaronis et al., available at this [link](https://jbiomedsem.biomedcentral.com/articles/10.1186/2041-1480-3-S1-S2).

## Usage  

To use the code, follow these steps:  

1. **Preprocess the data:**  
   Preprocess the data to handle qualifiers and synonyms.  
   ```bash  
   jupyter notebook Code/preprocessing.ipynb 
2. **Use pre-trained embeddings + classifiers pipeline:**  
   Use the provided scripts to train and evaluate the models on the preprocessed data. 

   Options for `model_name` are `BERT` (SciBERT), `T5_base`, or `T5_large`.    
   Options for `dataset_name` are `Omni`, `Emtree`, or `Mesh`.

   ```bash    
   python Code/pipeline.py --model_name BERT --dataset_name Omni --undersample_flag --undersample_ratio 0.1 --smote_flag --smote_ratio 0.1 --seed 42
3. **Fine-tune the transformer models:**    
   ```bash    
   python Code/finetune.py --model_name T5_base --dataset_name Omni --undersample_flag --undersample_ratio 0.1 --smote_flag --smote_ratio 0.1 --seed 42
## Examples  

Here are some examples of how to run the code:  

1. **Train classifiers with SciBERT embeddings on OmniScience dataset:**  
   ```bash  
   python Code/pipeline.py --model_name BERT --dataset_name Omni --undersample_flag --undersample_ratio 0.1  
2. **Fine-tune the T5_base model on the Mesh Dataset:**    
   ```bash    
   python Code/finetune.py --model_name T5_base --dataset_name Mesh --undersample_flag --undersample_ratio 0.1 

## Contributing
Contributions are welcome! If you have any suggestions or improvements, please create a pull request or open an issue.



## License  

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


## Contact  

For any questions or support, please contact:  

- **Author:** Dimitrios Papadopoulos  
- **Email:** dimitrios.papadopoulos@student.uva.nl
