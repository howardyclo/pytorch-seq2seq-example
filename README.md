Batched Seq2Seq Example
Based on the [`seq2seq-translation-batched.ipynb`](https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation-batched.ipynb) from *practical-pytorch*, but more extra features.

This example runs grammatical error correction task where the source sequence is a grammatically erroneuous English sentence and the target sequence is an grammatically correct English sentence. The corpus and evaluation script can be download at: https://github.com/keisks/jfleg.

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
- Replace unknown words by selecting the source token with the highest attention score. (Translation)

### Cons
Comparing to the state-of-the-art seq2seq library, OpenNMT-py, there are some stuffs that aren't optimized in this codebase:
- Use CuDNN when possible (always on encoder, on decoder when `input_feed`=0)
- Always avoid indexing / loops and use torch primitives.
- When possible, batch softmax operations across time. (this is the second complicated part of the code)
- Batch inference and beam search for translation (this is the most complicated part of the code)

### How to speed up RNN training?
Several ways to speed up RNN training:
- Batching
- Static padding
- Dynamic padding
- Bucketing
- Truncated BPTT 

See ["Sequence Models and the RNN API (TensorFlow Dev Summit 2017)"](https://www.youtube.com/watch?v=RIR_-Xlbp7s&t=490s) for understanding those techniques.

You can use torchtext or OpenNMT's data iterator for speeding up the training. It can be7x faster! (ex: 7 hour for an epoch -> 1 hour!)

### Acknowledgement
Thanks to the author of OpenNMT-py @srush for answering the questions for me! See https://github.com/OpenNMT/OpenNMT-py/issues/552
