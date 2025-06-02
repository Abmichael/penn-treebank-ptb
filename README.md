# Penn Treebank Language Modeling Project

## 🎯 Overview

This project implements neural language models on the Penn Treebank dataset, a standard benchmark for language modeling research. The project includes comprehensive data preprocessing, exploratory data analysis, and implementations of LSTM-based language models.

## ✨ Features

- **Complete Penn Treebank preprocessing pipeline**
- **Comprehensive exploratory data analysis** with Jupyter notebook
- **LSTM language model implementation** with PyTorch
- **Text generation capabilities**
- **Training and evaluation scripts**
- **Configuration-based experiment management**
- **TensorBoard logging support**

## 🚀 Quick Start

### 1. Setup Environment
```bash
pip install -r requirements.txt
```

### 2. Run Quick Demo
```bash
python demo_training.py
```

### 3. Full Training
```bash
python src/train.py --config config/config.yaml
```

### 4. Explore Data Analysis
```bash
jupyter notebook notebooks/ptb_exploratory_analysis.ipynb
```

## 📊 Dataset Statistics

| Split | Sentences | Words | Vocabulary | Avg Length |
|-------|-----------|-------|------------|------------|
| Train | 42,065 | 1,010,572 | 46,069 | 24.0 |
| Valid | 2,460 | 59,115 | 8,473 | 24.0 |
| Test | 2,346 | 55,851 | 8,441 | 23.8 |

## 🧠 Model Architecture

- **Type**: LSTM Language Model
- **Vocabulary**: 48,231 total words, 18,057 recommended (frequency ≥ 3)
- **Embedding Dimension**: 1024
- **Hidden Dimension**: 1024 
- **Layers**: 3
- **Parameters**: ~44M (full model), ~5.7M (demo model)
- **Features**: Tied embeddings, dropout regularization

## 📈 Performance

- **Initial Perplexity**: ~17,000 (untrained)
- **Demo Training**: 16,977 → 4,015 in 20 batches
- **Expected Final**: 80-120 perplexity on test set
- **Training Speed**: ~0.26s/batch on CPU

---

**Status**: ✅ Fully operational with validated training pipeline

**Demo**: Run `python demo_training.py` for quick showcase

## Project Structure

```
├── README.md
├── requirements.txt
├── data/
│   └── ptb/                    # Place PTB dataset files here
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Data preprocessing and loading
│   ├── model.py                # RNN model definition
│   ├── train.py                # Training script
│   ├── evaluate.py             # Evaluation script
│   └── utils.py                # Utility functions
├── config/
│   └── config.yaml             # Model and training configuration
├── notebooks/
│   └── exploratory_analysis.ipynb
└── scripts/
    └── download_data.py        # Instructions for obtaining PTB data
```

## Dataset

The Penn Treebank dataset (LDC99T42) is required for this project. You can obtain it from:
- [LDC Catalog: https://catalog.ldc.upenn.edu/LDC99T42](https://catalog.ldc.upenn.edu/LDC99T42)

The dataset contains:
- ~1 million words from Wall Street Journal articles
- Syntactic annotations and part-of-speech tags
- Train/validation/test splits commonly used for language modeling

## 🔧 Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Place PTB dataset files in the `data/ptb/` directory

## 📚 Usage

### Training
```bash
python src/train.py --config config/config.yaml
```

### Evaluation
```bash
python src/evaluate.py --model_path checkpoints/best_model.pth
```

### Project Cleanup
Before uploading to Git or Google Drive, clean up large temporary files:

**Python Script:**
```bash
# Preview what would be deleted
python scripts/cleanup_project.py --dry-run

# Standard cleanup (keeps model checkpoints)
python scripts/cleanup_project.py

# Aggressive cleanup (removes everything including models)
python scripts/cleanup_project.py --aggressive

# Interactive cleanup (ask before each deletion)
python scripts/cleanup_project.py --interactive
```

**Windows Users:**
```powershell
# PowerShell (recommended)
.\cleanup.ps1

# Or batch file
.\cleanup.bat
```

**Cleanup Benefits:**
- 🗂️ Removes extracted Penn Treebank data (saves ~150MB)
- 🧹 Clears Python cache and temporary files
- 📊 Removes TensorBoard logs and large log files
- ⚡ Significantly faster Git pushes and Google Drive uploads

**What's Preserved:**
- Source code (`src/`)
- Configuration files (`config/`)
- Compressed data archive (`data/ptb/*.tar.zst`)
- Processed text files (`data/ptb/*.txt`)
- Documentation and scripts

## Model Architecture

- Basic RNN (LSTM/GRU) with embedding layer
- Configurable hidden dimensions and layers
- Dropout for regularization
- Softmax output for next word prediction

## Performance Metrics

- Perplexity (primary metric for language modeling)
- Cross-entropy loss
- Accuracy on next word prediction

## References

- Marcus, M., Santorini, B., & Marcinkiewicz, M. A. (1993). Building a large annotated corpus of English: The Penn Treebank.
- Penn Treebank dataset: https://catalog.ldc.upenn.edu/LDC99T42
