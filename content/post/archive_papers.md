---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "Archive_papers"
subtitle: ""
summary: "Archive for regular papers"
authors: [admin]
tags: [Academic]
categories: [Computer Vision, Computer Graphics]
date: 2020-01-05T12:57:22Z
lastmod: 2020-01-05T12:57:22Z
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

## Deep Correlations for Texture Synthesis (TOG2017)
### Motivation
The texture synthesis using [deep feature maps](https://papers.nips.cc/paper/5633-texture-synthesis-using-convolutional-neural-networks.pdf) has difficulty to synthesize structure textures, which is a challenge to preserve the non-local structures.
### Core Ideas
1. Introduce a structure matrix to represent the deep correlation among deep features. In another word, [Gatys et al.,2015](https://papers.nips.cc/paper/5633-texture-synthesis-using-convolutional-neural-networks.pdf) consider the Gram loss of deep features between channels, while acutally the structure information is included inside each feature channel. Thus  [Sendik et al., 2017](https://docs.wixstatic.com/ugd/b1fe6d_f4f1684f6ba647ffbf1148c3721fdfc4.pdf) propose an intra-feature based Gram loss, which is fomulated as:

\begin{equation}
R_{i,j}^{l,n} = \sum_{q,m} w_{i,j} f_{q,n}^{l,n} f_{q-i,m-j}^{l,n}
\end{equation}
where $R_{i,j}^{l,n}$ denotes the intra-feature Gram loss of $n$th channel in $l$th layer, $i \in \[-Q/2, Q/2\]$ and $ j \in \[-M/2,M/2\]$.  This means that $f_{q,m}^{l,n}$ is shifted by $i$ pixels vertically and $j$ pixels horizontally, and applying a point-wise multiplication across the overlapping region, weighted by the inverse of the total amount of overlapping regions, That is
$$w_{i,j} = \[(Q-|i|)(M-|j|)\]^{-1}$
Then the structure loss based on intra-feature Gram loss is denoted as:

$$E_{DCor}^l = \frac{1}{4}\sum_{i,j,n}(R_{i,j}^{l,n} - \widetilde{R}_{i,j}^{l,n})^2$$

2. Introduce a diversity loss to make sure the method can synthesize a larger texture than input exemplar. The idea is to shift the input deep correlation matrix $f_{i,j}^{l,n}$ to the size of desired output texture.

3. Introduce an edge-preserving smooth loss, which only penalizes the pixel difference when none of neighbouring pixels are smiliar to the pixel under consideration. The authors claim that this smooth loss is useful to reduce checker artefacts.

4. The total loss function is weighted $E_{DCor}$, Gram loss, diversity loss and edge-preserving smooth loss.
