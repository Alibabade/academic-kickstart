---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "Activation_functions_in_dl"
subtitle: ""
summary: "Basic summary for understanding activation functions in NN"
authors: [admin]
tags: [Academic, Activation functions in DL]
categories: [Computer Vision, Deep Learning]
date: 2019-12-24T21:15:41Z
lastmod: 2019-12-24T21:15:41Z
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

## Activation functions

### Description
Activation functions works like an on-off button that determines whether the output of a neuron or what information should be passed to next layer. In biology, it works like synaptic in brain which decides what information it passes from one neuron cell to next one. There are several activation functions widely used in neural networks.
### Binary function (step function)
In one word, the output of binary function is 1 or 0 which is based on whether the input is greater or lower than a threshold. In math, it looks like this:
f(x) = {1, if x > T; 0, otherwise}. 

Cons: it does not allow multiple outputs, and it can not support to classify inputs to one of categories.

### Linear function
f(x) = $cx$. **Cons:** 1. the deviation of linear function is a constant, which does not help for backpropagation as the deviation is not correlated to its inputs, in another word, it can not distinguih what weights or parameters help to learn the task; 2. linear function makes the entire multiple neural network into one linear layer (as the combination of linear functions is still a linear function), which becomes a simple regression model. It can not handle complex tasks by varying parameters of inputs.

### Non-linear functions
Non-linear functions address the problems by two aspects:
1. The deviation of non-liear function is a function correlated to its inputs, which contributes the backpropagation to learn how to update weights for high accurancy.
2. Non-linear functions form the layers with hidden neurons into a deep neural network which is capable of predicting for complicated tasks by learning from complex datasets. 

There are several popular activation functions used in modern deep neural networks.
### Sigmoid/Logistic Regression
$$f(x) = \frac{1}{1+e^{-x}}$$

**Pros:** 
1. **smooth gradient**, no jumping output values compared to binary function.
2. **output value lies between 0 and 1**, normalizing output of each neuron.
3. **right choice for probability prediction**, the probability of anything exists only between 0 and 1.

**Cons:**
1. **vanishing gradient**, the gradient barely changes when $x>2$ or $x<-2$.
2. **computationally expensive**.

### Softmax
$$f(x_i)= \frac{x_i}{\Sigma_{j=1}^{n}{x_j}}$$

**Pros:** capable of handling multiple classification and the sum of predicted probabilities is 1. **Cons:** only used for output layer.

**Softmax is more suitable for multiple classification case when the predicted class must and only be one of categories. k-sigmoid/LR can be used to classify such multi-class problem that the predicted class could be multiple.** 

### Tanh


