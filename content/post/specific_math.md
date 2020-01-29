---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "Specific_math"
subtitle: ""
summary: "Archive for some specific and easy forgotten mathematic."
authors: [admin]
tags: [Academic, mathematic]
categories: [mathematic]
date: 2020-01-28T20:50:01Z
lastmod: 2020-01-28T20:50:01Z
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

This post is an archive for some specific and easy forgotten mathematic knowledge like linear algebra and optimization behind ML/DL.

## 1. Linear Algebra

### 1.1 Rank of a matrix
Given $n$ number of math functions with $m$ variables and it is written as following:

\begin{eqnarray}
&a_{11} x_1 + a_{12} x_2 + &a_{13} x_3 + &\cdot \cdot \cdot + a_{1m} x_{m} &= b_1 \\\\\\
&a_{21} x_1 + a_{22} x_2 + &a_{23} x_3 + &\cdot \cdot \cdot + a_{2m} x_{m} &= b_2 \\\\\\
&a_{31} x_1 + a_{32} x_2 + &a_{33} x_3 + &\cdot \cdot \cdot + a_{3m} x_{m} &= b_3 \\\\\\
&\cdot \cdot \cdot         &\cdot \cdot \cdot    &\cdot \cdot \cdot        &\cdot \cdot \cdot    \\\\\\
&a_{n1} x_1 + a_{n2} x_2 + &a_{n3} x_3 + &\cdot \cdot \cdot + a_{nm} x_{m} &= b_n \\\\\\
\end{eqnarray}

If we extract all the coefficient from the math functions above, and name each row of coefficient as $\alpha_i$ where $i=(1,2,...,n)$, then we have a column of vectors ($\alpha_1, \alpha_2, ..., \alpha_n$) and each of them is as following:

\begin{eqnarray}
  &(a_{11}, a_{12}, \cdot \cdot \cdot, a_{1m}) &= \alpha_1 \\\\\\
  &(a_{21}, a_{22}, \cdot \cdot \cdot, a_{2m}) &= \alpha_2 \\\\\\
  &\cdot\cdot\cdot                                         \\\\\\
  &(a_{n1}, a_{n2}, \cdot \cdot \cdot, a_{nm}) &= \alpha_n \\\\\\
\end{eqnarray}

Now we explain a noun called **linear independent set** consisting of $\alpha_i$. If each coefficient $k_i \in \mathbb{R}$ where $i=(1,2,...,s)$ have to be 0 like $k_1 = k_2 = \cdot \cdot \cdot = k_s = 0$ in order to make the following equation equal 0:
$$k_1 \alpha_1 + k_2 \alpha_2 + \cdot \cdot \cdot + k_s \alpha_s = 0$$
then we say $(\alpha_1,\alpha_2,...,\alpha_s)$ is a linear independent set.

**For a matrix, the Rank is the maximum number of $\alpha$ to be a linear independent set. In other words, Rank is the number of $\alpha$ in the maximal linear independent set.**

For example, only if $k_1$ and $k_2$ have to be 0 to make sure $k_1 \alpha_1 + k_2 \alpha_2=0$, then we say $(\alpha_1, \alpha_2)$ is a linear independent set, and the number of this linear independent set is 2. Next, only if $k_1=k_2=k_3=0$ makes sure $k_1 \alpha_1 + k_2 \alpha_2 + k_3 \alpha_3=0$, then we say $(\alpha_1, \alpha_2, \alpha_3)$ is a linear independent set, and the number is 3. Continue, we **firstly** find that **NOT** all of $k_1$,$k_2$,...,$k_s$ and $k_{s+1}$ have to be 0 and it still makes $k_1 \alpha_1 + k_2 \alpha_2 + k_3 \alpha_3 + \cdot \cdot \cdot + k_s \alpha_s=0$ happen, then we say $(\alpha_1, \alpha_2, \alpha_3,...,\alpha_{s+1})$ is not a linear independent set. And we call $(\alpha_1, \alpha_2, \alpha_3,...,\alpha_{s})$ as the maximal linear independent set, and the number is $s$. If each $\alpha_i$ contains one row of coefficient in a series of equations, then the Rank of the matrix composed by all the coefficient is $s$.

### 1.2 Inverse of a matrix
**If a matrix $A$ has $n$ rows and $n$ columns**, which is named $A_{n \times n}$, and rank of $A_{n \times n}$ is equal to $n$, then the matrix has an inverse matrix. The definition of inverse of a matrix is, there is a matrix $B_{n \times n}$, which makes sure the following equation:
$$AB=BA=I_n$$
where $I_n$ is the identity matrix (or unit matrix), of size $n \times n$ , with ones on the main diagonal and zeros elsewhere.
