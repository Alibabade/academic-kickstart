---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "Basic_understanding_dl"
subtitle: ""
summary: "Some discrete knowledge across research areas like NLP, IR, image/video and geometry"
authors: [admin]
tags: [Academic, Discrete knowledge in ML/DL]
categories: [ML/DL]
date: 2019-12-24T15:52:34Z
lastmod: 2019-12-24T15:52:34Z
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

## Bag of words (BOW)
BOW is a method to extract features from text documents, which is usually used in **NLP**, **Information retrieve (IR) from documents** and **document classification**. In general, BOW summarizes words in documents into a vocabulary (like dict type in python) that **collects all the words in the documents along with word counts but disregarding the order they appear.** 

For examples, two sentences: 
```python
Lei Li would like to have a lunch before he goes to watch a movie.
```

```python
James enjoyed the movie of Star War and would like to watch it again.
```
BOW will collect all the words together to form a vocabulary like:
```python
{"Lei":1, "Li":1, "would":2, "like":2, "to":3, "have":1, "a":2, "lunch":1, "before":1, "he":1, "goes":1, "watch":2, "movie":2, "James":1, "enjoyed":1, "the":1, "of":1, "Star":1, "War":1, "and":1, "it":1, "again":1 }
```
The length of vector represents each sentence is equal to the word number, which is 22 in our case.
Then first sentence is presented in vector (in the order of vocabulary) as: {1,1,1,1,2,1,2,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0}, and the second sentence is presented in vector as: {0,0,1,1,1,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1}.  

### Reference
https://www.freecodecamp.org/news/an-introduction-to-bag-of-words-and-how-to-code-it-in-python-for-nlp-282e87a9da04/

## Principal Component Analysis (PCA)
1. mean: $\mu_i = \frac{1}{n} \Sigma_{i=1}^n{x_i}$
2. variance: $\sigma^2 = \frac{1}{n} \Sigma_{i=1}^n{(x_i - \mu_i)^2}$ 
3. standard deviation: $\sigma^2 = \frac{1}{n-1} \Sigma_{i=1}^n{(x_i - \mu_i)^2}$
4. covariance: $cov(x,y) = \frac{1}{n-1} \Sigma_{i=1}^n{(x_i-\mu_x)*(y_i -\mu_y)}$



