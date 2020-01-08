---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "Graph_NN"
subtitle: ""
summary: "Basic summary of Graph Convolutional Networks"
authors: [admin]
tags: [Academic]
categories: [Computer Vision, DL]
date: 2020-01-05T17:26:13Z
lastmod: 2020-01-05T17:26:13Z
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
This blog simply clarifies the concepts of graph embedding, graph neural networks and graph convolutional networks.

## Graph Embedding
Graph Embedding (GE) is in representation learning of neural network, which often contains two types:
1. embed each node in a graph into a low-dimension, scalar and dense vector, which can be represented and inferenced for learning tasks.
2. embed the whole graph into a low-dimension, scalar and dense vector, which can be used for graph structure classification.

There are three types of method to complete graph embedding:
1. Matrix  Factorization. A representable matrix of a graph is factorized into vectors, which can be used for learning tasks. The representable matrix of a graph are often adjacency matrix, laplacian matrix etc.
2. Deepwalk. Inspired by word2vec, the deepwalk considers the reached node list by random walk as a word, which is fed into word2vec network to obtain a embeded vector for learning tasks.
3. Graph Neural Network. It is basically a series of neural networks operating on graphs. The graph information like representable matrix is fed into neural networks in order to get embedded vectors for learning tasks.

## Graph Neural Networks
Graph Neural Networks (GNN) are deep neural networks with graph information as input. In general, the GNN can be divided into different types: 1. Graph Convolutional Networks (GCN); 2. Graph Attention Networks (GAT); 3. Graph Adversarial Networks; 4. Graph LSTM. The basic relationship between GE, GNN,GCN is shown as following picture: 
{{< figure library="true" src="relation_GE_GNN_GCN.jpg" title="Fig 3. Relation between GE, GNN and GCN in [this blog](https://zhuanlan.zhihu.com/p/89503068)" lightbox="true" >}} 

## Graph Convolutional Networks
Graph Convolutional Networks (GCN) operate convolution on graph information like adjacency matrix, which is similar to convolution on pixels in CNN. To better clarify this concept, we will use equations and pictures in the following paragraph.

### Concepts
There are two concepts should be understood first before GCN.
**Degree Matrix (D)**: this matrix ($N \times N$, N is the node number) is a diag matrix in which values in diag line means the degree of each node; **Adjacency Matrix (A)**: this matrix is also a $N \times N$ matrix in which value $A_{i,j}=1$ means there is an edge between node $i$ and $j$, otherwise $A_{i,j}=0$; 

### Simple GCN example
Let's consider one simple GCN example, which has one GCN layer and one activation layer, the fomulation is as following:
$$f(H^{l}, A) = \sigma(AH^{l}W^{l})$$ 
where $W^l$ denotes the weight matrix in the $l$th layer and $\sigma(\dot)$ denotes the activation function like ReLU. This is the simplest expression of GCN example, but it's already much powerful (we will show example below). However, there are two basic limitations of this simple fomulation:
1. there is no node self information as adjacency matrix $A$ does not contain any information of nodeself.
2. there is no normalization of adjacency matrix. The fomulation $AH^{l}$ is actually a linear transformation which scales node feature vectors $H^l$ by summing the feaures of all neighbour nodes. The nodes having more neighbour nodes has more impact, which should be normalized.

**Fix limitation 1.** We introduce the identity matrix $I$ into adjacency matrix $A$ to add nodeself information. For example, $\hat{A} = A + I_n$ where $I_n$ is the identity matrix with $n \times n$ size.
**Fix limiation 2.** Normalizing $A$ means that all rows of $A$ should sum to be one, and we realize this by $D^{-1}A$ where $D$ is the diag degree matrix. In practise, we surprisingly find that using a symmetric normalization, e.g., $D^{-\frac{1}{2}}AD^{-\frac{1}{2}}$ is more 


## Reference
1. https://zhuanlan.zhihu.com/p/89503068
2. https://towardsdatascience.com/how-to-do-deep-learning-on-graphs-with-graph-convolutional-networks-62acf5b143d0
3. https://tkipf.github.io/graph-convolutional-networks/
