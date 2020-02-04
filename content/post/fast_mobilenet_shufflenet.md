---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "Fast_mobilenet_shufflenet"
subtitle: ""
summary: "Brief summary of efficient Mobilenet and Shufflenet"
authors: [admin]
tags: [Academic]
categories: [Computer Vision, DL]
date: 2020-02-03T21:50:00Z
lastmod: 2020-02-03T21:50:00Z
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
## 1. Background
In this post, we use the following symbols for all section:
1. $W$: width, $H$: height
2. $N$: input channel, $M$: output channel
3. $K$: convolution kernel size.


### 1.1 Params for a neural network
Params is related to the model size, the unit is Million in float32, thus the model size is approximately 4 times of params. For a standard convolution operation, the params = $(K^2 \times N + 1)M$, without bias: $K^2 \times NM$. For a standard fully connection layer, the params = $(N+1)M$, without bias: $NM$.

### 1.2 Computation complexity (FLOPs)
Computational complexity (or cost) is related to speed (but indirect), and it usually written as FLOPs. Here only multiplication-adds is considered. For a standard convolution operation, the FLOPs = $WHN \times K^2M$. For a fully


### 1.3 Compute the params and FLOPs in PyTorch
In pytorch, opCounter library can be used to compute the params and FLOPs of a model.
Install opCouter first:
```python
pip install thop
```
For instance, computing these numbers can be done by following code:
```python
from torchvision.models import resnet50
from thop import profile

model = resnet50()
input = torch.randn(1,3,224,224)
flops, params = profile(model, inputs=(input,))
```

## 2. Computational cost for convolution layers
{{< figure library="true" src="computation_conv.png" title="Fig 1. The illustration of computational cost in a convolution operation. Image source:[this blog](https://medium.com/@yu4u/why-mobilenet-and-its-variants-e-g-shufflenet-are-fast-1c7048b9618d)." lightbox="true" >}}

We will check on the computation of a general convolution operation. For example, in Fig 1, the input feature has the dimensions like width $W$ $\times$ height $H$ $\times$ N (input channels), the convolution kernel has dimension like $K \times K$ (kernel size) $\times$ M (output channels) and the convolutional operation has stride=1 and padding=1 which keeps the width and height same between input and output, thus the output feature will have the dimension $W \times H \times M$. Then the **multiply-add computation** (standard computational cost) of a general conv operation is $WHN \times K^2 M$.

### 2.1 Computation cost of conv $3 \times 3$ and conv $1 \times 1$
Normally, the most used conv kernel in modern neural network is $3 \times 3$, which is denoted as conv $3 \times 3$. Its computational cost is $WHN3^2M$ when the convolution operates on both spatial and channel domain. If we illustrate the computation cost on spatial and channel domain, the following fig could be a better visualization. We can there is a fully connection between input channels and output channels.
{{< figure library="true" src="computation_cost1.png" title="Fig 2. The illustration of computational cost for conv $3\times3$ and conv $1\times1$ operation. Image recreated from [this blog](https://medium.com/@yu4u/why-mobilenet-and-its-variants-e-g-shufflenet-are-fast-1c7048b9618d)." lightbox="true" >}}

For conv $1\times1$, the spatial connect is $1\times1$ linear projection while channel projection is still fully connected, which leads to computational cost $WHN \times 1^2 M$. Compared to conv $3 \times 3$, the computation is reduced by $\frac{1}{K^3} = \frac{1}{9}$ in this case.  

### 2.2 Computation cost of group convolution
Group convolution is firstly introduced in [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) to deal with the insufficient GPU memory and, to some extent, reduce the learned number of parameters. Grouped convolution operates on channel domain, which divides the channels into $G$ groups and the information in different groups is not shared. In each group, the connection still follows the fully connection way.
The computation costs for grouped conv $3\times3$ and conv $1\times1$ are as following:
{{< figure library="true" src="computation_cost2.png" title="Fig 3. The illustration of computational cost for grouped conv $3\times3$ where $G=2$ in this case. Image recreated from [this blog](https://medium.com/@yu4u/why-mobilenet-and-its-variants-e-g-shufflenet-are-fast-1c7048b9618d)." lightbox="true" >}}

Compared to standard conv, the grouped conv $3\times3$ (where $G=2$) reduce the connection in channel domain by factor $G$, which results in $\frac{1}{G}$ times of standard conv.

### 2.3 Computation cost of depthwise convolution
Depthwise convolution is firstly introduced in [MobileNet v1](https://arxiv.org/pdf/1704.04861.pdf), which performs the convolution operation independently to for each of input channels. Actually, this can be regarded as a special case of grouped conv when $G=N$. Usually, output channel $M >> K^2$, thus depthwise conv significantly reduces the computational cost compared to standard conv operation.
{{< figure library="true" src="computation_cost3.png" title="Fig 4. The illustration of computational cost for depthwise conv $3\times3$. Image recreated from [this blog](https://medium.com/@yu4u/why-mobilenet-and-its-variants-e-g-shufflenet-are-fast-1c7048b9618d)." lightbox="true" >}}

### 2.4 Channel shuffle
Channel shuffle is an operation introduced in [ShuffleNet v1](https://arxiv.org/pdf/1707.01083.pdf)

## 3. ResNet, ResNeXt

## 4. MobileNet v1 vs v2

## 5. ShuffleNet v1 vs v2

TO BE CONTINUED...
## Reference:
1. https://zhuanlan.zhihu.com/p/37074222
2. https://medium.com/@yu4u/why-mobilenet-and-its-variants-e-g-shufflenet-are-fast-1c7048b9618d
3. https://zhuanlan.zhihu.com/p/51566209
