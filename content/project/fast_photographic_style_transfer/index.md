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
Photographic style transfer aims to transfer only the colour information from a given reference image to a source image without detail distortions. However, neural style transfer methods (i.e., [Neural-Style](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf) and [Fast-Neural-Style](https://cs.stanford.edu/people/jcjohns/eccv16/)) usually tend to distort the details of source image to complete artistic transformation (including colours and textures) for reference images. Thus preserving details or structures in source images without affecting colour transformation is the key to photographic style transfer.

## Method
The idea behind this method is purely strightforward, which preserves the artistic style transformation but matches the colour distribution of reference images better to the source image. Fast-Neural-Style (or Neural-Style) tends to keep the structure details of source images on a single high-level conv layer, which distorts these details and transforms the textures (including colours) of reference images to unexpected regions in source images. In experiments, an interesting thing is found that simply **restricting the structure details of source images in multiple conv layers (both low-level and high level) is able to suppress texture (of reference image) expression and match better colour distribution on generated images**. However, this still causes the detail loss of source images. To address this problem, **a post-processing step is introduced to extract the detail information from original source image and transfer it to transformed images**. In image processing, an image is composed by its colour and detail information, in math, $I=C+D$ where $C$ and $D$ denotes the colour and detail information, respectively. Thus $D = I - C$ where $C$ is obtained from image smoothing technique like DTRF in this case.

In total, the proposed method consists of two steps:1. Fast-Neural-Style with multiple conv layers restriction on detail preservation and a similarity loss; 2, Post-processing Refinement with detail extraction and exchange to transformed image from step 1. The training stage and testing stage are illustrated in below figures:

{{< figure src="training.png" title="Fig 1. Training Stage." lightbox="true" >}}
{{< figure src="testing.png" title="Fig 2. Testing Stage." lightbox="true" >}}

## More results
Here are more stylized examples by this method:
{{< figure src="example1.png" lightbox="true" >}}
{{< figure src="example2.png" lightbox="true" >}}
{{< figure src="example3.png" lightbox="true" >}}
{{< figure src="example4.png" lightbox="true" >}}
{{< figure src="example5.png" lightbox="true" >}}
{{< figure src="example6.png" lightbox="true" >}}

## Limitations
1. The source image and reference image should share a similar semantic contents, otherwise the transformation will fail to generate faithful results as we do not apply semantic masks for input images.
2. This method works well for photography images which basically have 1-3 colour tones. To extreme colourful images, this approach usually fails to achieve faithful results.

## One more thing
The github code is released in [here](https://github.com/Alibabade/Fast-photographic-style-transfer).
