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

The encoder is composed by a bidirection RNN, a context vector is the sum of weighted hidden states and the decoder translates the context vector to a output target based on previous output targets.  

### Reference
1. https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html

