---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "Object_detection"
subtitle: ""
summary: "Summary of object detection in DL"
authors: [admin]
tags: [Computer Vision, DL]
categories: [Academic]
date: 2020-01-09T12:21:28Z
lastmod: 2020-01-09T12:21:28Z
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
This blog contains four parts:
1. Introduction: What is Object Detection? and general thoughts/ideas to deal with Object Detection;
2. Classic Deep Learning based Methods: multi-stage :RCNN and SPP Net , two-stage: Fast RCNN, Faster RCNN, Mask RCNN;
3. Classic One-Stage Methods: YOLOv1-v3, SSD, RetinaNet;
4. More Recent Anchor-Free Object Detection Methods (2018-2019);

## 1. Introduction
**What is Object Detection?** Given an image, object detection aims to find the categories of objects contained and their corresponding locations (presented as bounding-boxes) in the image. Thus Object Detection contains two tasks: classification and localization.

**General thoughts/ideas to detect objects.** The classification has been done by CNNs like AlexNet, VGG and ResNet. Then only localization still needs to be done. There are two intuitive ways: 1. Regression: the location of an object is presented by a vector $(x,y,w,h)$ which are the centre coordinates and width/height of an object bounding-box in the given image. For example, given an image, it only contains one object---cat. To locate the bounding-box, we apply a CNN to predict the vector $(x_p,y_p,w_p,h_p)$ and learn to regress the predicted vector to be close to groundtruth $(x_t,y_t,w_t,h_t)$ by calculating the L2 loss (see Fig 1.).   

{{< figure library="true" src="object_detection_regression1.png" title="Fig 1. Regression for localization shown in [this blog](https://blog.csdn.net/v_JULY_v/article/details/80170182)." lightbox="true" >}}

If the initial bounding-box is **randomly chosen** or **there are many objects**, then the entire regression will be much difficult and take lots of training time to correct the predicted vector to groundtruth. Sometime, it may not achieve a good convergence. However, if we select approximately initial box coordinates which probably contains the objects, then the regression of these boxes should be much easier and faster as these initial boxes already have closer coordinates to groundtruth than random ones. Thus, we could divide the problem into Box Selection and Regression, which are called **Region Proposal Selection** and **Bounding-box Regression**(please go post Basic_understanding_dl if you do not know Bounding-box regression) in Object Detection, respectively. Based on this, the early Object Detection methods contain multi-stage tasks like: Region Proposal Selection, Classification and Bounding-box Regression. In this blog, we only focus on the DL-based techniques, thus We do not review any pre-DL methods here.

There are a few candidate Region Proposal Selection methods (shown in below), and some of them are able to select fewer proposals (nearly hundreds or thousands) and keep high recall.
{{< figure library="true" src="Region_Proposal_Selections.png" title="Fig 2. Comparisons between different Region Proposal Selection methods shown in [this blog](https://blog.csdn.net/v_JULY_v/article/details/80170182)." lightbox="true" >}}
## 2. Classic Deep Learning based Methods
Since using Region Proposal Selection can reduce bounding-box candidates from almost infinite to ~2k for one image with multiple objects, [Ross et al. 2014](https://arxiv.org/pdf/1311.2524.pdf) propose the first CNN-based Object Detection method, which uses CNN to extract features of images, classifies the categories and regress bounding-box based on the CNN features.
### 2.1 R-CNN (Region CNN)
The basic procedure of R-CNN model:
1. Use **Selective Search** to select ~2k Region Proposals for one image.
2. **Warp** all the Region Proposals into a **same size** as the fully connection layers in their backbone neural network (i.e., AlexNet) has image size limitation. For example, the FC layers only take 21x21xC feature vector as input, then all the input image size has to be 227x227 if all the Conv + BN + relu layers of a pre-trained AlexNet are preserved.  
3. Feed the Region Proposals into the pre-trained AlexNet at **each proposal per time** rate, and extract the CNN features from FC7 layer for further **classification** (i.e., SVM).
4. The extracted CNN features will also be used for **Bounding-box Regression**.

Based on the procedure above, there are twice fine-tuning:
1. Fine-tune the pre-trained CNN for classification. For example, the pre-trained CNN (i.e., AlexNet) may have 1000 categories, but we may only need it to classify ~20 categories, thus we need to fine-tune the neural network.
2. Fine-tune the pre-trained CNN for bounding-box regression. For example, we add a regression head behind the FC7 layer, and we need to fine-tune the network for bounding-box regression task.

#### 2.1.1 Some common tricks used
1. Non-Maximum Suppression
Commonly, sometimes the RCNN model outputs multiple bounding-boxes to localize the same object in the given image. To choose the best matching one, we use non-maximum suppression technique to avoid repeated detection of the same instance. For example, we sort all the boxes with confidence score, and remove those with low confidence score. Then we choose the box with highest IOU.   
{{< figure library="true" src="non-max-suppression.png" title="Fig 3. Non-maximum suppression used in RCNN [this blog](https://blog.csdn.net/v_JULY_v/article/details/80170182)." lightbox="true" >}}

#### 2.1.2 Problems of RCNN
RCNN extracts CNN features for each region proposal by feeding each of them into CNN once at a time, and the proposals selected by Selective Search are approximately 2k for each image, thus this process consumes much time. Adding pre-processing Selective Search, RCNN needs ~47 second per image.

### 2.2 SPP Net (Spatial Pyramid Pooling Network)
To speedup RCNN, SPPNet focuses on how to fix the problem that each proposal is fed into the CNN once a time. The reason behind the problem is the fully connected layers need fixed feature size (i.e., 1 x 21 x 256 in [He et al.,2014](https://arxiv.org/pdf/1406.4729.pdf)) for further classification and regression. Thus SPPNet comes up with an idea that an additional pooling layer called spatial pyramid pooling is inserted right after the last Conv layer and before the Fc layers. The operation of this pooling first projects the region proposals to the Conv features, then divides each feature map (i.e., 60 x 40 x 256 filters) from the last Conv layer into 3 patch scales (i.e., 1,4 and 16 patches, see Fig 4. For example, the patch size is: 60x40 for 1 patch, 30x20 for 4 patches and 15x10 for 16 patches, next operates max pooling on each scaled patch to obtain a 1 x 21(1+4+16) for each feature map, thus we get 1x21x256 fiexd vector for Fc layers.

{{< figure library="true" src="SPPNet_spatial_pyramid_pooling_layer.png" title="Fig 4. The spatial pyramid pooling layer in [SPPNet](https://arxiv.org/pdf/1406.4729.pdf)." lightbox="true" >}}

By proposing spatial pyramid pooling layer, SPPNet is able to reuse the feature maps extracted from CNN by passing the image once through because all information that region proposals need is shared in these feature maps. The only thing we could do next is project the region proposals selected by Selective Search onto these feature maps (**How to project Region Proposals to feature maps? I still do not get it.**). This operation extremely saves time consumption compared to extract feature maps per proposal per forward (like RCNN does). The total speedup of SPPNet is about 100 times compared to RCNN.

## Fast RCNN

## Reference
1. https://blog.csdn.net/v_JULY_v/article/details/80170182
2. https://lilianweng.github.io/lil-log/2017/12/31/object-recognition-for-dummies-part-3.html
