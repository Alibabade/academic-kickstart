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

To address the flickering problem, a few approaches made their attempts to achieve coherent video transfer results. In early stage, [Anderson et al.](https://arxiv.org/pdf/1605.08153.pdf) and [Ruder et al.](https://arxiv.org/pdf/1604.08610.pdf) are the very first to introduce temporal consistency by optical flow into video style transfer, and they achieve high coherent results but along with worse ghosting artefacts. Besides, their methods need 3 or 5 mins for each video frame which is less practical in video applications. [Huang et al.](http://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Real-Time_Neural_Style_CVPR_2017_paper.pdf) and [Gupta et al.](http://openaccess.thecvf.com/content_ICCV_2017/papers/Gupta_Characterizing_and_Improving_ICCV_2017_paper.pdf) propose real-time video style transfer by combining [Fast-Neural-Style](http://svl.stanford.edu/assets/papers/JohnsonECCV16.pdf) and temporal consistency. More recently, [Chen et al.](https://arxiv.org/pdf/1703.09211.pdf) and [Ruder et al.](https://arxiv.org/pdf/1708.04538.pdf) propose their methods to achieve more coherent results but sacrifice speed.

## 2. Motivation
We notice that all the methods aforementioned above are built upon feed-forward networks which are sensitive to small perturbations among adjacent frames, for example, lighting, noises and motions may cause large variations in stylised video frames. Thus **there are still space to be improved**. Besides, their networks are all **in a per-network-per-style pattern**, which means a training process is needed for each style and the training time may range from hours to days. In contrary, optimisation-based approaches are more stable for perturbations and naturally made for arbitrary styles. Thus we follow the optimisation-based routine.

Now we need to deal with the problems such as slow runtime and ghosting artefacts. We dig into the reason behind these problems, and observe that there are two drawbacks of previous optimisation-based methods (e.g., [Anderson et al.](https://arxiv.org/pdf/1605.08153.pdf) and [Ruder et al.](https://arxiv.org/pdf/1604.08610.pdf)): **1. their methods complete the entire style transformation for each video frame, which causes 3 or 5 mins; 2. they have too much temporal consistency constraints between adjacent frames, which causes ghosting artefacts.** To avoid these drawbacks, we come up with a straightforward idea that we only constrain loose temporal consistency among already stylised frames.  

Following this idea, we need to handle another two problems: **1. inconsistent textures between adjacent stylised frames due to flow errors (ghosting artefacts); 2. image degeneration after long-term running (blurriness artefacts).**
## 3. Methodology
1. Prevent flow errors (ghosting artefacts) via multi-scale flow, incremental mask and multi-frame fusion.
2. Prevent image degeneration (blurriness artefacts) via sharpness loss consists of perceptual losses and pixel loss.
3. Enhance temporal consistency with loose constraints on both rgb-level and feature level.
## 4. Qualitative Evaluation
### 4.1 Ablation study
{{< figure src="11.png" title="Ablation study on proposed mask techniques." >}}
{{< figure src="4.png" title="Ablation study on proposed sharpness loss." >}}
### 4.2 Comparison to state-of-the-art methods
We compare our approach with state-of-the-art methods, and these experiments demonstrate that our method produces more stable and diverse stylised video than them.
{{< video src="3_ijcv.mp4" controls="yes" >}}
{{< video src="1_ijcv.mp4" controls="yes" >}}
## 5. More results

TO BE CONTINUED...
