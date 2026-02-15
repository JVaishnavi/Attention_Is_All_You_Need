# Attention Is All You Need: Transformer Implementation from Scratch

A complete, ground-up implementation of the original Transformer model proposed in the paper *"Attention Is All You Need"* (Vaswani et al., 2017). This repository contains a PyTorch implementation of the sequence-to-sequence Encoder-Decoder architecture, designed for Neural Machine Translation (NMT).

## 🚀 Key Features
* **Built from First Principles:** Every component (Self-Attention, Multi-Head Attention, LayerNorm, Positional Encoding) is implemented from scratch without using pre-built Transformer layers.
* **Efficient Training:** Implements the **One Cycle Learning Rate Policy** for faster convergence.
* **High Performance:** Achieves a **Cross Entropy Loss < 3.6 in under 10 epochs** on standard translation datasets (English-French / English-Italian).
* **Visualizations:** Includes attention map visualizations to interpret how the model attends to different parts of the input sentence.

---

## 🧠 Model Architecture

The Transformer replaces traditional Recurrent Neural Networks (RNNs) and CNNs with an architecture based entirely on attention mechanisms. This allows for significantly more parallelization and better handling of long-range dependencies.

<img width="920" height="1308" alt="image" src="https://github.com/user-attachments/assets/7e8fd27c-6b28-4785-8136-ffb713de6751" />


### The Encoder-Decoder Structure
The model consists of an **Encoder** that maps an input sequence of symbol representations $(x_1, ..., x_n)$ to a sequence of continuous representations $z = (z_1, ..., z_n)$. Given $z$, the **Decoder** then generates an output sequence $(y_1, ..., y_m)$ of symbols one element at a time.



### Core Concepts Implemented

#### 1. Positional Encoding
Since the model contains no recurrence and no convolution, we must inject some information about the relative or absolute position of the tokens in the sequence. We use sine and cosine functions of different frequencies:

$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})$$
$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})$$

#### 2. Multi-Head Attention
Instead of performing a single attention function, we linearly project the queries, keys, and values $h$ times with different, learned linear projections. This allows the model to jointly attend to information from different representation subspaces at different positions.

#### 3. Position-wise Feed-Forward Networks
Each of the layers in our encoder and decoder contains a fully connected feed-forward network, which is applied to each position separately and identically.

---

## 📉 Performance & Results

This implementation was tested on **English-to-French** and **English-to-Italian** translation tasks.

* **Training Speed:** Optimized using `OneCycleLR` scheduler.
* **Convergence:** Reached valid loss **< 3.6** within **10 epochs**.
* **Inference:** Supports greedy decoding and beam search.

| Metric | Value |
| :--- | :--- |
| **Final Validation Loss** | **< 3.6** |
| **Epochs to Converge** | **10** |
| **Optimizer** | **Adam** ($\beta_1=0.9, \beta_2=0.98, \epsilon=10^{-9}$) |


├── models/
│   ├── embedding.py       # Input Embeddings & Positional Encoding
│   ├── attention.py       # Scaled Dot-Product & Multi-Head Attention
│   ├── encoder.py         # Encoder Layer & Stack
│   ├── decoder.py         # Decoder Layer & Stack
│   └── transformer.py     # Full Transformer Assembly
├── training/
│   ├── train.py           # Training loop & Validation
│   └── optimizer.py       # Learning rate scheduling
├── utils/
│   ├── tokenizer.py       # Byte-Pair Encoding (BPE) / Word-Level
│   └── plotting.py        # Attention map visualization
├── notebooks/
│   └── Demo_Notebook.ipynb # Walkthrough & Visuals
└── README.md


## 📜 References
* Vaswani, A., et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762). NeurIPS.
* Annotated Transformer (Harvard NLP).

---

## 📂 Repository Structure
