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

### Cons
Comparing to the state-of-the-art seq2seq library, OpenNMT-py, there are some stuffs that aren't optimized in this codebase:
- Use CuDNN when possible (always on encoder, on decoder when input_feed 0)
- Always avoid indexing / loops and use torch primitives.
- When possible, batch softmax operations across time. ( this is the second complicated part of the code)
- Batch inference and beam search for translation (this is the most complicated part of the code)

Thanks to the author of OpenNMT-py @srush for answering the questions for me! See https://github.com/OpenNMT/OpenNMT-py/issues/552
