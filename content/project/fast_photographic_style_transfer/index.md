---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "Fast_photographic_style_transfer"
subtitle: ""
summary: ""
authors: [admin]
tags: [Academic]
categories: [Computer Vision, DL]
date: 2020-01-13T21:34:39Z
lastmod: 2020-01-13T21:34:39Z
featured: false
draft: false

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
# Focal points: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight.
image:
  caption: "Photographic Style Transfer"
  focal_point: "Center"
  preview_only: false

# Projects (optional).
#   Associate this post with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `projects = ["internal-project"]` references `content/project/deep-learning/index.md`.
#   Otherwise, set `projects = []`.
projects: []
---

This project aims to implement a torch version for fast photographic style transfer based on [Fast-Neural-Style](https://cs.stanford.edu/people/jcjohns/eccv16/). The teaser image is a stylized result by the algorithm described in this project, which takes around 1.40 seconds for $852 \times 480$ resolution image on a single NVIDIA 1080Ti card.

In this project, I also provide a torch implementation of the Domain Transform (Recursive Filter) which is described in the paper:

    Domain Transform for Edge-Aware Image and Video Processing
    Eduardo S. L. Gastal and Manuel M. Oliveira
    ACM Transactions on Graphics. Volume 30 (2011), Number 4.
    Proceedings of SIGGRAPH 2011, Article 69.


## Introduction
Photographic style transfer aims to transfer only the colour information from a given reference image to a source image without detail distortions. However, neural style transfer methods (i.e., [Neural-Style](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf) and [Fast-Neural-Style](https://cs.stanford.edu/people/jcjohns/eccv16/)) usually tend to distort the details of source image to complete artistic transformation (including colours and textures) for reference images. Thus preserving details or structures in source images withour affecting colour transformation is the key to photographic style transfer.

## Method
To be continued...
