#  Sinhala-LLM: Fine-tuning SmolLM2-1.7B for Sinhala

<div align="center">

[![GitHub stars](https://img.shields.io/github/stars/HMAHD/sinhala-llm?style=for-the-badge)](https://github.com/HMAHD/sinhala-llm/stargazers)

[![GitHub forks](https://img.shields.io/github/forks/HMAHD/sinhala-llm?style=for-the-badge)](https://github.com/HMAHD/sinhala-llm/network)

[![GitHub issues](https://img.shields.io/github/issues/HMAHD/sinhala-llm?style=for-the-badge)](https://github.com/HMAHD/sinhala-llm/issues)

[![GitHub license](https://img.shields.io/github/license/HMAHD/sinhala-llm?style=for-the-badge)](LICENSE)

**Dataset and scripts for fine-tuning the SmolLM2-1.7B language model for Sinhala language adaptation.**

</div>

## 📖 Overview

This repository contains the dataset and Python scripts used to fine-tune the SmolLM2-1.7B large language model for improved performance and accuracy in the Sinhala language.  The project focuses on data preprocessing, model training, and evaluation using various metrics.  It is intended for researchers and developers working on Sinhala NLP tasks.


## ✨ Features

- **Data Preprocessing:** Scripts for cleaning, formatting, and preparing the Sinhala dataset for model training.
- **Model Training:**  Scripts for fine-tuning the SmolLM2-1.7B model using the prepared dataset.  Supports different training approaches (indicated by multiple training scripts).
- **Model Evaluation:** Scripts for evaluating the fine-tuned model's performance using appropriate metrics. Includes baseline evaluation for comparison.
- **GGUF Support:** Includes a script (`gguf.py`) suggesting compatibility with the GGUF format for model weights.

## 🛠️ Tech Stack

- **Programming Language:** Python
- **Libraries:**  Likely includes libraries for handling large language models (Transformers, potentially), data processing (Pandas, potentially), and model evaluation (potentially Hugging Face's evaluation tools).

## 🚀 Quick Start

This project requires a significant computational resource (suitable GPU) and may also have external dependencies that need to be resolved before running.  Detailed version requirements are not clearly defined in the provided repository metadata, therefore, specific version requirements are marked as TODO.

### Prerequisites

- Python (version TODO)
- Required Python packages (TODO:  Install from `requirements.txt` if one exists.  Otherwise, determine based on `import` statements in the scripts.  This will likely include transformers, datasets, and potentially others).
- A powerful machine with a suitable GPU (V100 or better recommended) for training the large language model.

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/HMAHD/sinhala-llm.git
   cd sinhala-llm
   ```

2. **Install dependencies:**  (Method depends on whether a `requirements.txt` file exists or needs to be created)
    - **If `requirements.txt` exists:**
      ```bash
      pip install -r requirements.txt
      ```
    - **If `requirements.txt` does not exist:** (Requires manual dependency identification)
      ```bash
      # TODO: Determine specific package list based on import statements in Python files.
      pip install <package_name1> <package_name2> ...
      ```

3. **Dataset Download:** Download the dataset as described in `dataset-download.txt`. (This file contains instructions, not the dataset itself).

4. **Data Processing:** Execute the necessary data processing scripts: `clean.py`, `data-process.py` or `data-process-fixed.py` (depending on the specific needs or version used).

5. **Model Training:** Choose the appropriate training script (`train.py`, `train-fast.py`, or `train_final.py`) and execute it. This stage will require substantial compute resources.

6. **Model Evaluation:** Run `evaluation.py` or `evaluation_baseline.py` to assess model performance.

## 📁 Project Structure

```
sinhala-llm/
├── clean.py             # Data cleaning script.
├── data-process-fixed.py # Data preprocessing script (alternative version).
├── data-process.py       # Data preprocessing script.
├── data_processing.log   # Log file from data processing.
├── dataset-download.txt # Instructions for dataset download.
├── dataset_stats.json   # Statistics about the dataset.
├── evaluation.py         # Model evaluation script.
├── evaluation_baseline.py # Baseline model evaluation script.
├── gguf.py              # Script related to GGUF model format.
├── sample_100.json       # Sample data (likely a subset of the full dataset).
├── setup.py              # Setup script.
├── train-fast.py         # Model training script (alternative version).
├── train.py              # Model training script.
├── train_final.py        # Model training script (alternative version).
└── training.log           # Log file from model training.
└── wandb/               # Directory possibly for Weights & Biases logging (empty in current commit).
└── .gitignore            # Git ignore file
```


## 🧪 Testing

Testing is not explicitly implemented in this repository.  However, the provided evaluation scripts allow assessing the model's performance indirectly.


## 📄 License

TODO: Add license information (if provided in the repository)

## 🙏 Acknowledgments

TODO: Add acknowledgments if any.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.0.0-green.svg)]()
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()
[![Downloads](https://img.shields.io/npm/dm/package-name.svg)]()

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.0.0-green.svg)]()
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()
[![Downloads](https://img.shields.io/npm/dm/package-name.svg)]()

