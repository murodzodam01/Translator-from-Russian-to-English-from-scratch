# Russian-to-English Neural Machine Translation from Scratch

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Seq2Seq-EE4C2C?logo=pytorch&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-Machine%20Translation-6A5ACD)

## Overview

This project builds a **Russian-to-English neural machine translation system from scratch** using PyTorch. It implements the complete sequence-to-sequence pipeline: text preprocessing, Byte Pair Encoding, vocabulary construction, recurrent encoding, autoregressive decoding, masked training, attention, and BLEU evaluation.

Two architectures are compared:

1. A baseline GRU encoder–decoder that compresses the source sentence into a fixed-size representation.
2. An attention-enhanced GRU model that dynamically focuses on relevant source positions during decoding.

The attention model improves development BLEU from **17.33** to **24.95**, an absolute gain of **7.62 points**.

The full implementation, saved learning curves, and translation examples are available in [`Russian_English_Neural_Machine_Translation.ipynb`](Russian_English_Neural_Machine_Translation.ipynb).

## Motivation

Machine translation brings together several foundational NLP challenges:

- representing rare words and proper names;
- processing variable-length sequences;
- learning relationships between two vocabularies;
- generating output tokens autoregressively;
- retaining information from long source sentences;
- evaluating generated text objectively.

This project emphasizes implementation transparency. Rather than relying on a pretrained translation pipeline, it constructs the model components directly to show how information moves through an encoder–decoder system and how attention reduces the fixed-context bottleneck.

## Dataset

The parallel corpus contains Russian and English hotel descriptions. Each record consists of two tab-separated sentences:

```text
English sentence<TAB>Russian sentence
```

The notebook reserves **3,000 sentence pairs** for development evaluation using a fixed random seed (`random_state=42`). The remaining observations are used for training.

Hotel descriptions provide a useful translation domain because they contain common sentence structures alongside difficult names, locations, amenities, distances, and uncommon descriptive terms.

## Text preprocessing

### Tokenization

Text is lowercased and initially tokenized with NLTK's `WordPunctTokenizer`.

### Byte Pair Encoding

Separate Russian and English [Byte Pair Encoding](https://github.com/rsennrich/subword-nmt) rules are learned with **8,000 merge symbols per language**.

BPE offers a compromise between word- and character-level tokenization:

- frequent words can remain complete;
- rare words can be represented using known subword units;
- vocabulary size remains manageable;
- out-of-vocabulary problems are reduced.

### Vocabulary

The project uses a custom `Vocab` class to map subword tokens to integer indices and restore generated index sequences to text. Beginning-of-sequence, end-of-sequence, and padding tokens support batched training and autoregressive decoding.

## Model architectures

### Baseline GRU encoder–decoder

The baseline model contains:

- separate source and target embedding layers;
- a GRU encoder;
- a linear projection for initializing the decoder;
- a `GRUCell` decoder;
- a linear vocabulary projection;
- greedy autoregressive inference.

During training, teacher forcing supplies reference target tokens to the decoder. During inference, generation starts with the BOS token and repeatedly selects the most probable next token.

### Encoder–decoder with additive attention

The second model retains the complete encoder-state sequence. At every decoding step, additive attention calculates an alignment score for each source position:

$$a_t = W_o\tanh(W_eh_t^e + W_dh^d)$$

The scores are normalized into attention probabilities:

$$p_t = \frac{e^{a_t}}{\sum_\tau e^{a_\tau}}$$

The resulting context vector is a weighted sum of encoder states:

$$c = \sum_t p_t h_t^e$$

Padding positions are masked before softmax. The context vector is concatenated with the current target embedding and passed to the decoder GRU cell.

## Training

Both models use:

- Adam optimization;
- mini-batches of 32 sentence pairs;
- teacher forcing;
- masked token-level negative log-likelihood;
- development BLEU tracking during training.

| Model | Embedding size | Hidden size | Training iterations |
| --- | ---: | ---: | ---: |
| Baseline GRU | 64 | 128 | 25,000 |
| Attention GRU | 256 | 256 | 10,000 |

The sequence mask includes the first EOS token but excludes later padding positions. Loss is normalized by the number of valid target tokens in each sequence.

## Results

| Model | Development BLEU | Improvement |
| --- | ---: | ---: |
| Baseline GRU encoder–decoder | 17.33 | — |
| GRU encoder–decoder with attention | **24.95** | **+7.62** |

The attention model produces a substantial improvement even though it is trained for fewer iterations. Qualitative examples show better handling of sentence structure, amenities, and common hotel-description phrases.

The remaining errors are concentrated around:

- proper nouns and place names;
- rare subword combinations;
- exact property and landmark names;
- long sentences containing multiple clauses;
- repeated or partially omitted phrases.

These patterns are consistent with the limitations of compact recurrent translation models trained on a specialized corpus.

## Example

**Russian input**

```text
кроме того, предоставляется прокат велосипедов, услуги трансфера и бесплатная парковка.
```

**Attention-model translation**

```text
the property also offers bike hire, shuttle service and free parking.
```

## Repository structure

```text
.
├── README.md
├── Russian_English_Neural_Machine_Translation.ipynb
├── vocab.py
└── data.txt
```

- `Russian_English_Neural_Machine_Translation.ipynb` contains the complete experiment.
- `vocab.py` provides vocabulary encoding, masking, and text reconstruction.
- `data.txt` contains the tab-separated English–Russian parallel corpus.

## Installation

Clone the repository and install the dependencies:

```bash
git clone <your-repository-url>
cd <repository-name>
pip install numpy matplotlib nltk scikit-learn torch tqdm subword-nmt jupyter
```

Start Jupyter:

```bash
jupyter notebook Russian_English_Neural_Machine_Translation.ipynb
```

## Data path

The notebook currently reads the corpus from:

```python
/content/data.txt
```

This path is suitable for Google Colab after uploading `data.txt` to the session. For local execution, change it to a repository-relative path:

```python
data_path = "data.txt"
```

Make sure `vocab.py` is in the same directory as the notebook before running:

```python
from vocab import Vocab
```

## Key technologies

- Python
- PyTorch
- NumPy
- NLTK
- scikit-learn
- Matplotlib
- Byte Pair Encoding
- GRU encoder–decoder networks
- Additive attention
- BLEU evaluation

## Potential improvements

- Add beam-search decoding instead of greedy generation.
- Use a bidirectional or multilayer encoder.
- Visualize attention matrices for individual translations.
- Introduce dropout, learning-rate scheduling, and gradient clipping.
- Share BPE rules across both languages.
- Evaluate performance by sentence length and rare-token frequency.
- Compare the from-scratch models with a pretrained multilingual transformer.
- Add chrF or COMET alongside BLEU for a broader quality assessment.

## References

- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)
- [BLEU: a Method for Automatic Evaluation of Machine Translation](https://aclanthology.org/P02-1040/)
