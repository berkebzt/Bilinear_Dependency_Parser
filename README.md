
# Neural Bilinear Dependency Parsing

This project implements a neural dependency parser using a bilinear attention mechanism, based on a bidirectional LSTM encoder. The model is trained and evaluated on Universal Dependencies data in `.conllu` format. Designed for educational and research use, this notebook includes all components from data preprocessing to training, evaluation, and sample predictions.

## 🧠 Objective

The goal is to predict syntactic head-dependency relationships in a sentence using a deep learning model that combines word embeddings, LSTM layers, and bilinear scoring for dependency arcs.

## 📁 Files

- `Neural_Biliniear_Dependency_Parsing.ipynb`: Main notebook containing the entire pipeline: data loading, model definition, training, evaluation, and prediction.
- `en_ewt-ud-train.conllu`: (Expected input) Universal Dependencies file used for training.

## 🧰 Dependencies

```bash
torch
conllu
tqdm
sklearn
```

All dependencies can be installed using:

```python
!pip install torch conllu tqdm
```

## 🚀 Key Components

### 📦 Data Handling

- Custom `DependencyDataset` class parses `.conllu` files and converts tokens to indices.
- Vocabulary builder dynamically constructs word-index mappings.
- Data split into Train/Validation/Test subsets.

### 📊 Model Architecture

- **Embedding Layer**: Converts word indices into dense vectors.
- **BiLSTM Encoder**: Captures contextual representations.
- **MLPs for Head/Dependent**: Projects LSTM outputs to head and dependent spaces.
- **Bilinear Layer**: Scores each word pair (i, j) for likelihood of j being the head of i.

### ⚙️ Training & Evaluation

- Training loop with PyTorch’s `CrossEntropyLoss` ignoring padded labels.
- Accuracy-based evaluation on test set.
- Sample prediction loop for qualitative analysis of results.

### 📈 Output

Prints:
- Epoch-wise training loss.
- Final test accuracy.
- Example sentences with predicted vs. true head indices.

## 📌 Usage Instructions

1. Place your `.conllu` file in Google Drive.
2. Update the `data_file` path in the notebook.
3. Run all cells to train and evaluate the model.

## 🧪 Example Output

```text
Sentence: ['The', 'cat', 'sat', 'on', 'the', 'mat']
True Heads: [1, 2, -1, 2, 5, 3]
Predicted Heads: [1, 2, -1, 2, 5, 3]
```

## 🎓 Academic Context

This project may be part of an NLP or deep learning coursework module, illustrating the construction and training of a neural dependency parser from scratch.

---

## ⚠️ Disclaimer

This is a simplified academic implementation. Real-world parsing tasks would benefit from advanced architectures like Biaffine or Transformers and external embeddings (e.g., BERT).
