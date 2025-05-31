# Penn Treebank Language Modeling

## 1. Introduction

This document provides a comprehensive overview of the concepts, methods, and techniques employed in the Penn Treebank (PTB) Language Modeling project. The primary goal of this project is to build and evaluate neural language models capable of predicting the next word in a sequence, trained on the widely-used Penn Treebank dataset.

## 2. Core Concepts

### 2.1. Language Modeling (LM)
Language Modeling is a fundamental task in Natural Language Processing (NLP) that involves assigning probabilities to sequences of words. A good language model assigns a higher probability to a grammatically correct and semantically plausible sentence than to a nonsensical one.
Mathematically, for a sequence of words $W = w_1, w_2, ..., w_n$, the model estimates $P(W) = P(w_1, w_2, ..., w_n)$. This is often decomposed using the chain rule of probability:
$P(W) = \prod_{i=1}^{n} P(w_i | w_1, ..., w_{i-1})$

### 2.2. Neural Language Models
Traditional language models often relied on N-grams. Neural Language Models (NLMs) use neural networks (e.g., RNNs, LSTMs, Transformers) to learn distributed representations (embeddings) of words and capture long-range dependencies in text, often outperforming N-gram models.

### 2.3. Perplexity (PPL)
Perplexity is the primary evaluation metric for language models. It measures how well a probability model predicts a sample. A lower perplexity score indicates that the language model is better at predicting the sample text. It is defined as the exponentiated average negative log-likelihood of a sequence:
$PPL(W) = P(w_1, w_2, ..., w_N)^{-\frac{1}{N}} = \exp\left(-\frac{1}{N} \sum_{i=1}^{N} \log P(w_i | w_1, ..., w_{i-1})\right)$
where N is the number of words in the test set.

### 2.4. Word Embeddings
Word embeddings are dense vector representations of words in a lower-dimensional space. They capture semantic relationships between words, such that words with similar meanings are closer in the vector space. This project utilizes an embedding layer to convert input word tokens into these dense vectors.

### 2.5. Tied Embeddings
Tied input and output embeddings (or tied weights) is a technique where the word embedding matrix (used to convert input words to vectors) and the output projection matrix (used to convert the model's hidden state to word probabilities) share the same weights. This significantly reduces the number of parameters in the model, especially for large vocabularies, and can improve performance.

## 3. Dataset: Penn Treebank (PTB)

### 3.1. Overview
The Penn Treebank (PTB) dataset, specifically LDC99T42, is a widely used benchmark for language modeling. It consists of Wall Street Journal (WSJ) articles.
- **Source**: Linguistic Data Consortium (LDC)
- **Content**: ~1 million words from WSJ articles, annotated with part-of-speech (POS) tags and syntactic structure.

### 3.2. Data Preprocessing
A crucial step in this project was the preprocessing of the raw PTB data to make it suitable for language modeling.
1.  **Extraction**: The original `LDC99T42_Penn_Treebank_3.tar.zst` archive was extracted.
2.  **Format Conversion**: The POS-tagged format was converted to plain text.
3.  **Splitting**: Standard train/validation/test splits were created using common section conventions:
    *   Training: Sections 02-21 (e.g., `ptb.train.txt`)
    *   Validation: Section 22 (e.g., `ptb.valid.txt`)
    *   Test: Section 23 (e.g., `ptb.test.txt`)
4.  **Sentence Boundary Markers**: `<eos>` (end-of-sentence) tokens were added to mark sentence boundaries. This helps the model learn sentence structure.
5.  **Vocabulary Creation**: A vocabulary was built from the training data.
    *   Words with a frequency below a certain threshold (e.g., < 3) were replaced with an `<unk>` (unknown) token.
    *   Special tokens like `<pad>` (for padding sequences), `<unk>`, and `<eos>` were included.
    *   The recommended vocabulary size is around 30,000 words.
6.  **Numericalization**: Text data was converted into sequences of integer IDs based on the vocabulary.

The primary script for this is `scripts/preprocess_ptb.py`.

### 3.3. Dataset Statistics
(As per `PROJECT_STATUS.md` and `README.md`)

| Split | Sentences | Words     | Vocabulary | Avg Length |
|-------|-----------|-----------|------------|------------|
| Train | 2,000     | 1,025,863 | 23,782     | 25.9       |
| Valid | 100       | 60,017    | 5,949      | 31.0       |
| Test  | 100       | 56,924    | 5,833      | 28.5       |

*Note: Vocabulary counts here might refer to unique words before applying frequency-based filtering for the final model vocabulary.*

### 3.4. Vocabulary Strategy
- **Recommended Vocab Size**: 30,000 words (based on frequency ≥ 3).
- **Coverage**: This covers ~99.1% of training tokens.
- **OOV Rate**: Expected ~2.8-2.9% on validation/test sets.
- **Special Tokens**: `<pad>`, `<unk>`, `<eos>`.

## 4. Exploratory Data Analysis (EDA)
A comprehensive EDA was performed using the `notebooks/ptb_exploratory_analysis.ipynb` notebook. Key analyses include:
- Dataset statistics and visualizations.
- Vocabulary analysis (size, distribution).
- Sentence length distribution (informing optimal sequence length for models).
- Word frequency analysis (often following Zipf's law).
- Out-of-Vocabulary (OOV) rate analysis.
- Recommendations for language modeling based on these findings.

## 5. Model Architecture

The primary model implemented is an LSTM-based Recurrent Neural Network (RNN).

### 5.1. LSTM (Long Short-Term Memory)
LSTMs are a type of RNN designed to overcome the vanishing gradient problem, allowing them to learn long-range dependencies in sequences.
- **Core Components**: LSTMs use a system of gates (input, forget, output) and a cell state to control the flow of information.
- **Structure**: The model typically consists of:
    1.  **Embedding Layer**: Maps input word indices to dense embedding vectors.
        *   Dimension: e.g., 512 or 1024 (as per `PROJECT_STATUS.md` and `README.md`).
    2.  **LSTM Layers**: One or more LSTM layers process the sequence of embeddings.
        *   Hidden Dimension: e.g., 1024.
        *   Number of Layers: e.g., 3.
    3.  **Dropout**: Applied between LSTM layers and/or after the embedding layer for regularization.
        *   Rate: e.g., 0.3.
    4.  **Output Layer (Linear/Dense)**: Maps the LSTM's final hidden state to a probability distribution over the vocabulary (logits).
        *   Tied with the input embedding layer for parameter efficiency.
    5.  **Softmax**: Converts logits to probabilities.

The model definition can be found in `src/model.py`.

### 5.2. Key Hyperparameters (from `PROJECT_STATUS.md` and `config.yaml`)
- **Sequence Length**: 70 tokens (chosen to cover ~90% of sentences).
- **Embedding Dimension**: 512 / 1024.
- **Hidden Dimension**: 1024.
- **Number of LSTM Layers**: 3.
- **Dropout Rate**: 0.3.

## 6. Training Techniques

The training pipeline is implemented in `src/train.py`.

### 6.1. Data Loading (`src/data_loader.py`)
- **PyTorch `Dataset` and `DataLoader`**: Used for efficient data handling, batching, and shuffling.
- **Batching**: Data is fed to the model in batches.
    *   Batch Size: e.g., 64.
- **Sequence Preparation**: Long sequences of text are typically broken down into fixed-length input and target sequences for training. For example, for a sequence length of 70, the model predicts the 71st word given the first 70.

### 6.2. Loss Function
- **Cross-Entropy Loss**: Standard loss function for classification tasks, including predicting the next word from a vocabulary. It measures the difference between the predicted probability distribution and the true distribution (one-hot encoded target word).

### 6.3. Optimization
- **Optimizer**: Adam or SGD with momentum are common choices.
    *   Learning Rate: e.g., 2e-4, often with a warmup schedule.
- **Learning Rate Scheduling**: Adjusting the learning rate during training (e.g., reducing it when validation performance plateaus) can help convergence.
- **Gradient Clipping**: Prevents exploding gradients (a common issue in RNNs) by capping the norm of the gradients.
    *   Clipping Value: e.g., 1.0.

### 6.4. Regularization
- **Dropout**: Randomly sets a fraction of input units to 0 at each update during training time to prevent co-adaptation of neurons and reduce overfitting.

### 6.5. Training Loop
- **Epochs**: One full pass through the entire training dataset.
- **Batches**: Training data is processed in mini-batches.
- **Forward Pass**: Input data is fed through the model to get predictions.
- **Loss Calculation**: The loss function compares predictions to actual target words.
- **Backward Pass (Backpropagation)**: Gradients of the loss with respect to model parameters are computed.
- **Optimizer Step**: Model parameters are updated based on the gradients.

### 6.6. Monitoring and Checkpointing
- **Validation Loop**: After each epoch (or periodically), the model is evaluated on the validation set to monitor performance (e.g., perplexity) and check for overfitting.
- **Model Checkpointing**: Saving the model weights (especially the best performing ones on the validation set) during training. Checkpoints are typically saved in a `checkpoints/` directory.
- **Early Stopping**: Training is stopped if the validation performance does not improve for a certain number of epochs (patience) to prevent overfitting and save computation.
    *   Patience: e.g., 5.
- **TensorBoard Logging**: Training metrics (loss, perplexity) and potentially model graphs or embeddings can be logged for visualization in TensorBoard.

## 7. Evaluation (`src/evaluate.py`)

### 7.1. Metrics
- **Perplexity**: The primary metric, calculated on the validation and test sets.
    *   Expected Baseline LSTM: PPL ~120-150.
    *   Expected Optimized LSTM: PPL ~80-120.
    *   Expected Transformer: PPL ~70-100 (if implemented).

### 7.2. Text Generation
- The trained language model can be used to generate new text sequences.
- This typically involves:
    1.  Providing a starting prompt (a word or sequence of words).
    2.  Feeding the prompt to the model to get a probability distribution for the next word.
    3.  Sampling a word from this distribution (e.g., greedy sampling, temperature-based sampling).
    4.  Appending the sampled word to the sequence and repeating the process.
- The script `src/evaluate.py` includes functionality for text generation (e.g., `--generate --num_samples 5`).

## 8. Project Workflow & Tools

### 8.1. Configuration Management
- **`config/config.yaml`**: A central YAML file stores hyperparameters and settings for the model, training, and data. This allows for easy experiment management and reproducibility.
- Different configurations (e.g., `config_colab_full.yaml`, `config_colab_quick.yaml`) can be used for different environments or experiment scales.

### 8.2. Key Scripts
- **`scripts/download_data.py`**: Provides instructions or utilities for obtaining the PTB dataset.
- **`scripts/preprocess_ptb.py`**: Handles the initial preprocessing of the raw PTB data.
- **`scripts/cleanup_project.py`**: A utility script to remove temporary files, cached data, and large extracted datasets to prepare the project for version control or sharing.
- **`src/train.py`**: Main script for training the language model.
- **`src/evaluate.py`**: Script for evaluating a trained model and generating sample text.
- **`src/model.py`**: Defines the neural network architecture(s).
- **`src/data_loader.py`**: Contains PyTorch `Dataset` and `DataLoader` implementations for PTB.
- **`src/utils.py`**: Contains utility functions used across the project (e.g., for logging, saving/loading models).
- **`demo_training.py`**: A script for a quick demonstration of the training pipeline, often on a smaller subset of data or for fewer epochs.

### 8.3. Analysis and Development Environment
- **Jupyter Notebooks**: `notebooks/ptb_exploratory_analysis.ipynb` is used for EDA.
- **Python**: The primary programming language.
- **PyTorch**: The deep learning framework used for model implementation and training.
- **Other Libraries**:
    - `numpy`: For numerical operations.
    - `yaml` (PyYAML): For parsing configuration files.
    - `nltk`, `spacy` (potentially for tokenization or advanced preprocessing, though PTB is often pre-tokenized).
    - `matplotlib`, `seaborn`: For plotting in EDA.
    - `tensorboard` / `tensorboardX`: For logging and visualizing training progress.

## 9. Technical Stack Summary
- **Programming Language**: Python 3.x
- **Deep Learning Framework**: PyTorch
- **Core Libraries**: NumPy, SciPy
- **Configuration**: YAML
- **Data Analysis & Visualization**: Jupyter, Pandas, Matplotlib, Seaborn
- **Utilities**: `tqdm` (for progress bars)

## 10. Learning Objectives Achieved (as per `PROJECT_STATUS.md`)
- Real-world dataset preprocessing and analysis.
- Statistical analysis and visualization skills.
- Understanding of neural language models.
- Proficiency in PyTorch implementation (data loaders, model architecture).
- Experiment design, configuration management, and reproducibility.
- Model evaluation techniques (perplexity) and interpretation.
- Code organization and documentation best practices.

## 11. References (from `PROJECT_STATUS.md` and `README.md`)
- **Penn Treebank**: Marcus, M., Santorini, B., & Marcinkiewicz, M. A. (1993). Building a Large Annotated Corpus of English: The Penn Treebank.
- **Language Modeling**: Bengio, Y., Ducharme, R., Vincent, P., & Jauvin, C. (2003). A Neural Probabilistic Language Model.
- **LSTM**: Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory.
- **Transformer**: Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. *Advances in neural information processing systems, 30*.
- **PTB LDC Catalog**: [https://catalog.ldc.upenn.edu/LDC99T42](https://catalog.ldc.upenn.edu/LDC99T42)

