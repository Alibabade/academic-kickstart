---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "Decomposition_for_normal_images"
summary: ""
authors: []
tags: []
categories: []
date: 2023-05-11T14:00:11+08:00

# Optional external URL for project (replaces project detail page).
external_link: ""

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
# Focal points: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight.
image:
  caption: "Decomposition for normal images"
  focal_point: "Center"
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

This project presents a Matlab program implemented to decompose a normal image into a structure normal image and a detail normal image, where a structure normal image contains the main structure information from the original normal image (obtained by applying DTRF smoothing filter), and a detail normal image contains the detail information from the original normal image.

## Introduction
Normal Decomposition is mainly used for manipulating normal information stored in normal images, where normal images are more like a 2.5D data, a bridge that CONNECTs 3D information and 2D images. Thus, geometry processing applications especially 3D bas-relief modelling is naturally appealing to use such images to reconstruct 3D infromation. However, bas-relief modelling generally has to cope with compression cases, thus it is an essential capability for such application to preserve geometry details well under compression circumstances.

{{< figure src="example1.png" title="Fig 1. Comparison between value subtraction and vector subtraction." lightbox="true" >}}
People dealing with geometry detail preservation, in general, consider to decompose details from normal images first, then manipulate them before reconstruction. However, detail information obtained by value subtraction is broken (see zoom-in details of \(e\) in Fig 1.), where value subtraction is operated on pixel-level from original normal images and smoothed normal images. In fact, the detail information should be a vector subtraction as the RGB colour of each pixel in a normal image represents a vector. Therefore, to obtain more correct and intact detail information, we should use vector subtraction instead of value subtraction. 

## Script in Matlab
[Here]() is a script written in Matlab to decompose normal based on vector subtraction. Additionally, the script also includes a smoothing function that applies DTRF filter to obtain a structure normal image by smoothing the origial normal image.


## Results
Here is a comparsion between detail information (see \(b\) and \(d\) in Fig 1.) and reconstruction results (see \(c\) and \(f\) in Fig 1.) between value subtraction and vector subtraction.
