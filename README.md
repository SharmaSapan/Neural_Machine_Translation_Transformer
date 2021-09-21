# Neural_Machine_Translation_Transformer
Neural Machine Translation in this example is referenced from "Attention is all you need" paper by Google Brain Team. 

Transformer architectures have allowed major improvements on Natural language processing tasks
and their success is strongly noticed in many machine learning benchmarks. Using only attention
mechanism instead of Recurrent based architectures have reduced many bottlenecks, provided
parallelization potential and it also avoids catastrophic forgetting. Transformer was introduced in
paper “Attention is all you need” by google brain team in 2017. Since then most famous models
that are emerging in the field of Natural Language Processing and other sequence to sequence
based problems consists of many transformers or its variants combined in different ways. In the
“Attention is all you need” paper, the authors used a sequence-to-sequence network for neural machine
translation that transforms a given sequence of elements such as sequence of the words in a sentence
into another sentence. The transformer model, which is entirely based on positional encoding and
attention, replaced the recurrent layers. It is commonly used in encoder to decoder architectures with
multi-headed self-attention layers.

<img src="https://user-images.githubusercontent.com/54603828/134092707-c96f945f-3401-447f-a308-792af02e5da2.png" width="400" height="500" />

The architecture was created using Tensorflow library and a nightly build to get updated library.
Dataset was imported from Google’s tensorflow datasets library. The data-set have 51785 training,
1193 validation, and 1803 test examples containing Portuguese and English translations. Batches of
64 were created to train and test on the data-set. A pre-trained Bert language representation was used
to feed vectors to the transformers with a vocabulary size of 5000.

The transformer architecture had 2 stacked layer of each decoder and encoder, 2 heads for attention,
the embedding dimensions were 64, which is input dimensions and remain same in the subsequent
layers and the units for inner feed forward layer is 256. An Adam optimizer is used with a variable
learning rate as specified in the "Attention is all you need paper". A residual dropout layer was used
to the output of each sub layer with a rate of 0.1

BLEU scores were for the test set using 300 examples comparing the predicted translation with the
original translation. Run with 2 epochs produced a result of four different weights(1,0.5,0.3,0.25) as
BLEU-1 results: 0.127 BLEU-2 results: 0.356 BLEU-3 results: 0.539 BLEU-4 results: 0.597
Comparisons using manual sentence translation was also done but due to very less accuracy number
which was because of limited computation had sub-par results. An accuracy of 30 percent was
achieved in 2 epochs but given more compute power it can easily achieve great results as shown by
authors of the paper.

Referenced paper achieved BLEU scores better than any previous state-of-the-art models on Englishto-
german and english to french tests at fraction of training cost compared to other due to parallel
nature of transformer architecture. It outperformed by more than 2.0 BLEU and reached a new 28.4
score, with a configuration of big transformer, 6 stacks, 16 heads, 1024 embedding dimension, 0.3
dropout layer using 300,000 steps completed in 4 days on a parallel P100 GPUs of 8.
