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

If we extract all the coefficient from the math functions above, and name each row of coefficient as $\alpha$, then we have a column of ($\alpha_1, \alpha_2, ..., \alpha_n$) as following:

\begin{eqnarray}
\begin{cases}
  &(a_{11}, a_{12}, \cdot \cdot \cdot, a_{1m}) &= \alpha_1 \\\\\\
  &(a_{21}, a_{22}, \cdot \cdot \cdot, a_{2m}) &= \alpha_2 \\\\\\
  &\cdot\cdot\cdot                                         \\\\\\
  &(a_{n1}, a_{n2}, \cdot \cdot \cdot, a_{nm}) &= \alpha_n \\\\\\
\end{cases}
\end{eqnarray}
