# Fully Batched PyTorch Seq2Seq Example
Based on the [`seq2seq-translation-batched.ipynb`](https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation-batched.ipynb) from *practical-pytorch*, but more extra features.

### Extra features
- Cleaner codebase
- Very detailed comments for learners
- Implement Pytorch native dataset and dataloader for batching
- Correctly handle the hidden state from bidirectional encoder and past to the decoder as initial hidden state.
- Fully batched attention mechanism computation (only implement `general attention` but it's sufficient). Note: The original code still uses for-loop to compute, which is very slow.
- Support LSTM instead of only GRU
- Shared embeddings (encoder's input embedding and decoder's input embedding)
- Pretrained Glove embedding
- Fixed embedding
- Tie embeddings (decoder's input embedding and decoder's output embedding)
- Tensorboard visualization
- Load and save checkpoint
