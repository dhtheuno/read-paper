# Longformer: The Long-Document Transformer

Iz Beltagy et al. \
Allen Institute for AI

*** 

<img src="https://user-images.githubusercontent.com/47840814/106617932-60599600-65b2-11eb-9c0e-46daf0b2e738.png" height= 300, width=600>

- How do we actually deal with long sequence document? 
- Usually BERT or other typical NLP model have a limitation on their input token length (512 or 1024)
- we can increase the input sequence but it requires lot of computation when we calcualte self attention ($O(n^2)$) 
- Longformer's attention mechanism is a drop-in replacement for the standard self-attetion and combines a local windowed attention with a task motivated global attention. ($O(n \times w)$)

***
## Introduction
- Recent work for the long document approaches preimarily focus on ``autoregressive language modeling``, while the application of long document transformers to document level NLP tasks in ``the transfer learning setting`` has remained largely unexplored. 
- Longformer will be the combination of 
    1. a windowed local-context self-attention
    2. an end task motivated ``global attention`` that encodes inductive bias about the task 

***
## Related Work
<img src="https://user-images.githubusercontent.com/47840814/106621467-f347ff80-65b5-11eb-8369-d60079a51637.png" height= 300, width=600>

### Long-Document Transformers 
- The model with most similar attention pattern is [Sparse Transformer](https://arxiv.org/abs/1904.10509)
    - dilated sliding window of blocks of size 8x8 provided by BlockSparse
- Longformer Requires a custom CUDA kernel, but it is more flexible and maintainable than BlockSparse which is implemented in C++, and designed for a specific version of TensorFlow

### Task-speicific Models for Long Documents
- The simplest ways to deal with the long documents were
    1. simply trucnates the document, comonly used for classification
    2. process each chunch separately than combines the activations with a task specific model
    3. multihop and open domain QA tasks uses two-stage model where the first state retrieves relevant documents that are passed onto the second stage for answer extraction
-  The some of the similar approaches were 
    1. [ETC](https://www.aclweb.org/anthology/2020.emnlp-main.19/) which used local + global attention in Transformers but it uses relative position embeddings and introduced new training objective (CPC loss) for pre-training, and configures global attention in a slightly different way.
    2. [BigBird](https://arxiv.org/abs/2007.14062) which was the extension of the ETC

*** 
## Longformer
<img src="https://user-images.githubusercontent.com/47840814/106621291-c398f780-65b5-11eb-901e-cc2e00c65a15.png" width=600>

- The transformer's computational complexity is from self-attention
- If we have n long sequence, $O(n^2)$ time on self-attention component
- Various types of attention pattern will be used to reduce the time/computational complexity on the self attention mechanism for long sequence document

### Attention Pattern
#### Sliding Window
- Longformer uses fixed-size window attention surrounding each token
- Multiple stacked layers of windowed attention will create large receptive field which top layers have 
    1. access to all input locations
    2. have the capacity to build representations that incorporate information across the entire input.
- The computational complexity of this patter is $O(n \times w)$ if window size is w
#### Dilated Sliding Window
- Dilation like in CNN 
- It will help attention window to go further than the simple sliding window
- Since transformer have multiple self attention head, it helps to have ``no dilation`` on some head to focus on ``local context`` and have one to focus on ``longer context`` 
#### Global Attention
- MLM approaches relies on the local context to predict masked token
- However, BERT style modern NLP task model needs special token which represents whole sequence
    1. BERT uses CLS token for classification
    2. For QA, model has to seperate question and passage 
- By allowing special tokens to attends all the tokens on the sequence, it will solve the problem
- But we also have to allow all the tokens to attend special token 
#### Linear Projections for Global Attention
- Calculate sliding window attnetion sepeartely and calculate score
- Global attnetion will be initialized with values that match sliding window attention

### Implmentation
- Current pytorch/tensorflow do not have a form of banded matrix multiplication
- Created custom CUDA kernel implementation using TVM (Appendix A for more detail)

***
## Autoregressive Language Modeling
- One of the ``fundamental`` tasks in NLP
- Recent prior work on modeling long sequences using transformers has relied on this tasks as their primary evaluation

### Attention Pattern
- Used dilated sliding widnow attention
- Different window sizes across the layers
    - Small in lower layer -> capture local information
    - Increase window size as move to higher layers -> learn higher-level representation of the entier sequence

- No dilated sliding window on lower layers to maximaize the capacity to learn and utlize the immedicate local context
- Higher layer has small amount of increasing dilation on on 2 heeads
    - Model attends to distant tokens without sacrificing local context
### Experiment Setup 
- Character-level LM 
    - text8 
    - enwik8
#### Training 
- Increased the window size of the model during each training phase
    - Seems like model needs more gradient updates to learn the local context first before learning to utilize longer context
- Train the moodel over 5 total phases with starting sequence length of 2048 and ending sequence length of 23040 on the last phase
### Results 
<img src="https://user-images.githubusercontent.com/47840814/106628047-be8b7680-65bc-11eb-9c13-45e79e0544fe.png" height= 300, width=600>\
<img src="https://user-images.githubusercontent.com/47840814/106628080-c9dea200-65bc-11eb-93cd-fcdb20da6101.png" height= 300, width=600>

***
## Pretraining and Finetunning
- Goal was to make a BERT like model (SOTA systems which are suitable for many NLP tasks) for a long document tasks
- Six tasks including
    - classification
    - QA
    - Coreference resolution.
- Resulting model can take up to 4,096 token
- Pretrained with ``MLM`` (masked language modeling)
- Pretraing from scratch would be too expensive so continued pretraining from [RoBERTa](https://arxiv.org/abs/1907.11692) released checkpoint with only making minimal changes to support Longformer's attention mechanism

#### Attention Pattern
- Sliding window with 512 size 
    - same amount of computation as RoBERTa
    - Adding dialation on a few attention head hurt performance, because it is not compatible with the pretrained RoBERTa weights -> Retraining model from scratch probably improved performance 

#### Position Embeddings
- RoBERTa has 512 position embedding
- To deal with 4096 long token, do not initialize position embedding randomly but copy RoBERTa's position embedding multiple times (8 time for this case)
- This is effective than what we could expect
<img src="https://user-images.githubusercontent.com/47840814/106639640-85590380-65c8-11eb-8e71-8cae3fff6e50.png" height= 300, width=600>

#### Continued MLM Pretraining
- Pretrained using fairseq with a corpus of long documents (Appendix C for detail)
- train two models: base and large

#### Frozen RoBERTa Weights
- Trained the model while freezing all RoBERTa weights and only training the new position embeddings.
    - BPC of 1.850 (down from 1.957 at initialization) but higher than when all trained

***
## Tasks
<img src="https://user-images.githubusercontent.com/47840814/106632616-4d01f700-65c1-11eb-9afe-e283b5864f11.png">

<img src="https://user-images.githubusercontent.com/47840814/106632789-791d7800-65c1-11eb-88ea-f9673374e923.png" height= 300, width=600>

<img src="https://user-images.githubusercontent.com/47840814/106633130-d6b1c480-65c1-11eb-9e54-31d604814e3f.png" height= 300, width=600>

<img src="https://user-images.githubusercontent.com/47840814/106633224-efba7580-65c1-11eb-8a7d-5a9f2ae32259.png" height= 300, width=600>

***
## Longformer-Encoder-Decoder(LED)
- How about long document seq2seq generation model?
- It uses local+global attention pattern of the Longformer on the encoder but uses full self-attention to the entire encoded toekns and to previously decoded locations.
- Used BART and followed BART's exact architecture in terms of number of layers and hidden sizes
- Extended position embedding to 16k tokens and initialize the new postion embedding matrix by repeatedly copying BART's 1k postion embeddings 16 times 
- Used arXiv summarization dataset which 90th percentile of document lengths is 14.5K tokens
- 1024 windw sized and global attention on the first s token

<img src="https://user-images.githubusercontent.com/47840814/106634555-3a88bd00-65c3-11eb-8d68-92a6654dd67a.png" height= 300, width=600>

<img src="https://user-images.githubusercontent.com/47840814/106634591-42e0f800-65c3-11eb-8c0b-b31e89734b2c.png" height= 300, width=600>