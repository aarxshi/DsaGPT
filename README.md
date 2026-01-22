# DSAGPT - A Minimal GPT-Style Transformer Built from Scratch

DSAGPT is a small transformer based language model built from scratch in PyTorch to explore how attention based models work in practice. The project focuses on understanding how a GPT style transformer is structured and trained by implementing each component explicitly, rather than relying on pretrained models or high-level frameworks that can hide many of the underlying details. The model is trained on a small dataset of DSA style question answer pairs and is designed to run quickly on modest hardware, making it suitable for learning, experimentation, and rapid iteration.

## Data and Tokenization

The dataset consists of simple DSA style question–answer text.

A basic **whitespace tokenizer** is used to keep preprocessing transparent:

- Text is split on whitespace
- A vocabulary is constructed directly from the dataset
- Two mappings are created:
  - token → index (`stoi`)
  - index → token (`itos`)
- Index `0` is reserved for padding

This simple tokenizer makes it easy to trace how text is converted into tokens and fed into the model.


## Model Architecture

DSAGPT follows a standard decoder only transformer structure similar to GPT models.

The main components are:

- **Token embeddings**, which map each token to a learned vector  
- **Positional embeddings**, which encode the order of tokens in a sequence  
- **Multi-head self attention**, implemented with causal masking so the model only attends to past tokens  
- **Feed forward layers**, applied independently at each position  
- **Residual connections and layer normalization**, used to stabilize training  

All of these components are implemented directly in PyTorch to make the data flow through the model explicit.



## Training

The training loop is implemented manually:

- Batches are sampled from the tokenized text using a fixed context window  
- The model is trained using cross-entropy loss for next-token prediction  
- AdamW is used as the optimizer, with gradient clipping for stability  
- Training and validation loss are monitored periodically  

The goal is not to achieve state of the art performance, but to observe and understand the training behaviour of a transformer model. Instead, the focus is on architectural clarity and training dynamics.



## Text Generation

Text generation is implemented autoregressively:

- Tokens are generated one at a time
- The context is limited to a fixed block size
- Temperature scaling and optional top-k sampling control randomness

This allows direct inspection of how generation changes with different parameters.



## Example Output

```
Q: What is a binary search tree?
A: A BST is a binary tree in which the left child contains smaller values and the right child contains larger values.
```

This example shows that the model has learned both the structure and basic content of the training data.



## Conclusion
By building a GPT-style transformer from scratch, this project provides a clear view of how tokenization, attention, and autoregressive generation work together in practice. The lightweight setup makes it easy to experiment with and understand the behaviour of transformer models without requiring extensive computational resources.
This project prioritizes:
- clarity over complexity  
- explicit implementations over abstractions  
- fast iteration on small datasets  
