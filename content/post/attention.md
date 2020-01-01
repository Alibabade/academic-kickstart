---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "Attention"
subtitle: ""
summary: "Basic summary about attention"
authors: [admin]
tags: [Academic, attention in dl]
categories: [Computer Vision, Deep Learning]
date: 2019-12-28T22:03:15Z
lastmod: 2019-12-28T22:03:15Z
featured: false
draft: false

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
# Focal points: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight.
image:
  caption: ""
  focal_point: ""
  preview_only: false

# Projects (optional).
#   Associate this post with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `projects = ["internal-project"]` references `content/project/deep-learning/index.md`.
#   Otherwise, set `projects = []`.
projects: []
---

## Attention
The attention mechanism was created to simulate the human visual attention on images or understanding attention on texts. It was firstly born for solving a problem that widely exists in natural language processing (NLP) models like seq2seq, which NLP models often tend to forget the first part of processed sentences. 

### Seq2seq model
The encoder-decoder architecture commonly used in Seq2seq model:
1. encoder, compress the input sentence into a context vector in a fixed length way which is regarded as a representation of the meaning of input sentence. 
2. decoder, fed by the context vector and translate the vector to output. In some early works, the last state of encoder is usually used as the initial state of decoder.  

Both of the encoder and decoder are recurrent neural networks, using LSTM or GRU units.

**The critical problem of seq2seq model.** The seq2seq model often forgets the first part of a long sentence once it completes translation from the entire sentence to a context vector. To address this problem, the [attention mechanism](https://arxiv.org/pdf/1409.0473.pdf) is proposed.

## The attention mechanism
The new architecture for encoder-decoder machine translaion is as following:
{{< figure library="true" src="attention_encoder-decoder-attention.png" title="Fig 1. The encoder-decoder architecture in Bahdanau et al. 2015" lightbox="true" >}}
The encoder is composed by a bidirection RNN, a context vector is the sum of weighted hidden states and the decoder translates the context vector to a output target based on previous output targets.  

### Fomula
Let **x**=$\(x_1, x_2,...,x_n\)$ denote the source sequence of length $n$ and **y**=$\(y_1, y_2,...,y_m\)$ denote the output sequence of length $m$, $\overrightarrow{h_i}$ denotes the forward direction state and $\overleftarrow{h_i}$ presents the backward direction state, then the hidden state for $i$th input word is fomulated as:
$$h_i = \[\overrightarrow{h_i}^T; \overleftarrow{h_i}^T\], i=1,2,...,n$$

The hidden states at position $t$ in decoder includes previous hidden states $s_{t-1}$, previous output target $y_{t-1}$ the context vector $c_t$, which is denoted as $s_{t} = f(s_{t-1}, y_{t-1}, c_{t})$, where the context vector $c_{t}$ is a sum of encoder hidden states of input sequence, weighted by alignment scores. For output target at position $t$, we have:
$$c_{t} = \sum_{i=1}^{n} \alpha_{t,i} h_i$$ 

$$ \alpha_{t,i}= align\(y_t, x_i\) = \frac{exp(score(s_{t-1},h_i))}{\Sigma^n_{i=1} exp(score(s_{t-1},h_{i}))} $$
The score $\alpha_{t,i}$ is assigned to the pair $\(y_t, x_i\)$ of input at position $i$ and output at position $t$, and the set of weights ${\alpha_{t,i}}$ denotes how much each source hidden state matches for each output. In [Bahdanau et al. 2015](https://arxiv.org/pdf/1409.0473.pdf), the score $\alpha$ is learnt by a feed-forward network with a single hidden layer and this network is jointly learnt with other part of the model. Since the score is modelled in a network which also has weight matrices (i.e., $v_a$ and $W_a$) and activation layer (i.e., tanh), then the learning function is fomulated as:
$$score(s_t, h_i) = v_a^T tanh(W_a\[s_t;h_i\])$$

## Self-attention
Self-attention (or intra-attention) is such an attention mechnaism that assigns correlation in a single sequence for an effective representation of the same sequence. It has been shown to be very useful in machine reading, abstractive summarizatin or image description generation. 

In [Cheng et al., 2016](https://arxiv.org/pdf/1601.06733.pdf), an application of self-attention mechanism is shown in machine reading. For example, the self-attention mechanism enables the model to learn a correlation between the current word and the previous part of the input sentence.
{{< figure library="true" src="self-attention.png" title="Fig 2. An example of self-attention mechanism in Cheng et al., 2016" lightbox="true" >}}

## Soft and Hard Attention
In image caption generation, the attention mechanism is applied and shown to be very helpful. [Xu et al.,2015](http://proceedings.mlr.press/v37/xuc15.pdf) shows a series of attention visualization to demonstrate how the model learn to summarize the image by paying attention to different regions.
{{< figure library="true" src="soft-attention.png" title="Fig 2. The visulation of attention mechanism for image caption generation in Xu et al., 2015" lightbox="true" >}}

The soft attention and hard attention is telled by whether the attention has access to the entire image or only a patch region:

**Soft** attention: the alignment weights are assigned to all the patches in the source image, which is the same type used in [Bahdanau et al. 2015](https://arxiv.org/pdf/1409.0473.pdf)
Pro: the model is smooth and differentiable
Con: computationally expensive when the source image is large

**Hard** attention: the alignment weights are only assigned to a patch in the source image at a time
Pro: less computation at the inference time
Con: the model is non-differentiable and requires more complicated techniques such as variance reduction and reinforcement learning to train.([Luong et al., 2015](https://arxiv.org/pdf/1508.04025.pdf))

### Reference
1. https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html

