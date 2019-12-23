---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "Normalization_in_DL"
subtitle: ""
summary: "Some basic summary to understand normalizaion in DL"
authors: [admin]
tags: [Academic, Neual Network Optimization--normalizaion]
categories: [Computer Vision, Deep Learning]
date: 2019-12-23T11:26:06Z
lastmod: 2019-12-23T11:26:06Z
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

## Batch Normalization

### Short Description

Batch Normalization (BN) is a basic method to initialize inputs to neural networks, which is described first in [Loffe et al. 2015]. In the early years, the neural network is sensitive to the hyperparameters, which makes it difficult to train and stabilize. To address this problem, Loffe et al. proposed a novel normalization method to accelerate the neural network trianing process.

### Training Problems

**1. Slow learning speed.**
The $W$ weights and $b$ bias (parameters) in each layer are updated along with each SGD iterative optimization (back propagation), which makes the distribution of inputs to the next layer changes all the time. Thus the learning speed becomes slow as each layer has to adapt to input changes.

**2. Slow convergence speed.**
For saturating nonlinearities like Sigmoid and Tanh, more and more inputs to activation functions may lay in the saturation regions along with the $W$ and $b$ increase. This further causes the gradient becomes close to 0 and weights update in a slow rate. 

These problems are described as **Internal Covariate Shift** in Loffe et al. 2015.
### Solutions to Internal Covariate Shift

**1. Whitening (machine learning).**
Whitening aims to linearly transform the inputs to zero mean and unit variances, which decorrelates the inputs. There are normally two whitening methods: PCA and ZCA, where PCA whitening transforms inputs to zero mean and unit variance, and ZCA whitening transforms inputs to zero mean and same variance as original inputs.

**2. Batch Normalization.**
Motivation: 1. whitening costs too much computation if it is put before each layer; 2. the distribution of transformed inputs are not as representative as original inputs.

Solutions: 
1. simplify the linearly transformation by following equations:
$$ \mu_j = \frac{1}{m}\Sigma_{i=1}^{m}{x_j^i}$$
$$\sigma^2_j = \frac{1}{m}\Sigma_{i=1}^{m}{(x_j^i - \mu_j)^2}$$
$$x_j^{i\'} = \frac{x_j^i - \mu_j}{\sqrt{\sigma_j^2 + \varepsilon}}$$
where $j$ denotes the $j$th layer, $\mu_j$ and $\sigma_j^2$ denote the mean and variance (standard deviation) of inputs $x_j$. $x_j^{i\'}$ denotes the transformation output which has zero mean and unit variance, $m$ denotes the sample number in $x_i$ and $\varepsilon$ ($\varepsilon=10^{-8}$) prevents the zero in variance. 

2. learn a linear transformation with parameter $\gamma$ and $\beta$ to restore original input distribution by following equation:
$$x_j^{i\"} = \gamma_j x_j^{i\'} + \beta_j$$  
where the transformed output will have the same distribution of original inputs when $\gamma_j^2$ and $\beta_i$ equal to $\sigma_j^2$ and $\mu_j$.

**Normalization in testing stage.** Generally, there may be just one or a few examples to be predicted in testing stage, the mean and std computed from such examples could be baised. To address this problem, the $\mu_{batch}$ and $\sigma^2_{batch}$ of each layer are stored to compute the mean and std for testing stage. For example, $\mu_{test} = \frac{1}{n} \Sigma \mu_{batch}$ and $\sigma^2_{batch}=\frac{m}{m-1} \frac{1}{n} \Sigma \sigma^2_{batch}$, then $BN(x_{test})=\gamma \frac{x_{test} - \mu_{test}}{\sqrt{\sigma^2_{test} + \varepsilon}} + \beta$.

### Advantages of BN
1. Faster learning speed due to stable input distribution.
2. Saturating nonlinearities like Sigmoid and Tanh can still be used since gradients are prevented from disappearing.
3. Neural network is not sensitive to parameters, simplfy tuning process and stabilize the learning process.
4. BN partially works as regularization, increases generalization ability. The mean and variance of each mini-batch is different from each other, which may work as some noise input for the nerual network to learn. This has same function as Dropout shutdowns some neurons to produce some noise input to the neural networks.

### Disadvantages of BN
1. NOT well for small mini-batch as the mean ans std of small mini-batch differ great from other mini-batches which may introduce too much noise into the NN training.
2. NOT well for recurrent neural network as one hidden state may deal with a series of inputs, and each input has different mean and std. To remember these mean and std, it may need more BN to store them for each input.
3. NOT well for noise-sensitive applications such as generative models and deep reinforcement learning. 

### Reference:
https://arxiv.org/pdf/1502.03167.pdf

https://zhuanlan.zhihu.com/p/34879333

## Weight Normalization

### Short description
Weight normalization is designed [Salimans et al. 2016] to address the disadvantages of BN which are that BN usually introduces too much noise when the mini-batch is small and the mean and std of each mini-batch is correlated to inputs. It eliminates the correlations by normalizing weight parameters directly instead of mini-batch inputs.

### Motivation and methodology
The core limitaion of BN is that the mean and std is correlated to each mini-batch inputs, thus a better way to deal with that is design a normalization without the correlation. To achieve this, Salimans et al. proposed a Weight Normalization which is denoted as:
$$w = \frac{g}{||v||} v$$
where $v$ denotes the parameter vector, $||v||$ is the Euclidean norm of $v$, and $g$ is scalar. This reparameterization has the effect of fixing the Euclidean norm of weight vector $w$, and we now have $||w||=g$ which is totally independent from parameter vector $v$. This operation is similar to divide inputs by standard deviation in batch normalization.

The mean of neurons still depends on $v$, thus the authors proposed a 'mean-only batch normalization' which only allows the inputs to subtract their mean but not divided by std. Compared to std divide, the mean subtraction seems to introduce less noise.

## Layer Normalization

### Short description
The layer normalization is inspired by batch normalization but designed to small mini-batch cases and extend such technique to recurrent neural networks. 

### Motivation and methodology
As mentioned in short description. To achieve the goal, the authors [Hinton et al. 2016] alter the sum statistic of inputs from batch dimension to feature dimension. For example, in a mini-batch (containing multiple input features), the computation can be described as following picture:
<div align = 'center'>
  <img src='static/img/layer_normalization.png' height="225px">
  <br>
</div>

As can be seen, batch normalization computes the sum statistic of inputs across the batch dimension while layer normalization does across the feature dimension. The computation is almost the same in both normalization cases, but the mean and std of layer normalization is independent of other examples in the same mini-batch. Experiments show that layer normalization works well in RNNs.
 
