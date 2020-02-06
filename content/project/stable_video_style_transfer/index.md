---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "Stable_video_style_transfer"
summary: ""
authors: [admin]
tags: [Academic]
categories: [Computer Vision, DL]
date: 2020-02-06T21:35:55Z

# Optional external URL for project (replaces project detail page).
external_link: ""

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
# Focal points: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight.
image:
  caption: ""
  focal_point: ""
  preview_only: false

# Custom links (optional).
#   Uncomment and edit lines below to show custom links.
# links:
# - name: Follow
#   url: https://twitter.com
#   icon_pack: fab
#   icon: twitter

url_code: ""
url_pdf: ""
url_slides: ""
url_video: ""

# Slides (optional).
#   Associate this project with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides = "example-slides"` references `content/slides/example-slides.md`.
#   Otherwise, set `slides = ""`.
slides: ""
---
This project aims to deal with the **flickering problem** caused by naively applying per-frame stylization methods (e.g., [Fast-Neural-Style](http://svl.stanford.edu/assets/papers/JohnsonECCV16.pdf) and [AdaIN](https://arxiv.org/pdf/1703.06868.pdf)) on videos.

## 1. Background
In 2016, [Gatys et al.](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf) are the first to propose an image style transfer algorithm using deep neural networks, which is capable of transforming artistic style (e.g., colours, textures and brush strokes) from a given artistic image to arbitrary photos. The visual appealing results and elegant design of their approach motivate many researchers to dig in this field which is called Neural Artistic Style Transfer by followers. Along with the speedup (nearly real-time) of similar methods, researchers gradually turn their focus to video applications. However, **naively applying these per-frame styling methods causes bad flickering problem which reflects on inconsistent textures among video adjacent frames.**

To address the flickering problem, a few approaches made their attempts to achieve coherent video transfer results. In early stage, [Anderson et al.](https://arxiv.org/pdf/1605.08153.pdf) and [Ruder et al.](https://arxiv.org/pdf/1604.08610.pdf) are the very first to introduce temporal consistency into video style transfer, and they achieve high coherent results but along with worse ghosting artefacts. 

## 2. Motivation

## 3. Methodology

## 4. Results

## 5. One more thing

TO BE CONTINUED...
