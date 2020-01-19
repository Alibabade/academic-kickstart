---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "Object_detection"
subtitle: ""
summary: "Summary of object detection in DL"
authors: [admin]
tags: [Academic]
categories: [Computer Vision, DL]
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
**1. Non-Maximum Suppression**

Commonly, sometimes the RCNN model outputs multiple bounding-boxes to localize the same object in the given image. To choose the best matching one, we use non-maximum suppression technique to avoid repeated detection of the same instance. For example, we have a set $B$ of candidate boxes, and a set $S$ of corresponding scores, then we choose the best box by following steps: 1. sort all the boxes with the scores, and remove the box $M$ with highest score from $B$, and add to set $D$; 2. check any box $b_i$ left in $B$, if the IoU of $b_i$ and $M$, remove $b_i$ from $B$; 3. repeat 1-2 until $B$ is empty. The box in $D$ is what we want.      
{{< figure library="true" src="non-max-suppression.png" title="Fig 3. Non-maximum suppression used in RCNN [this blog](https://blog.csdn.net/v_JULY_v/article/details/80170182)." lightbox="true" >}}

**2. Hard Negative Mining**

Bounding-box containing no objects (i.e., cat or dog) are considered as negative samples. However, not all of them are equally hard to be identified. For example, some samples purely holding background are "easily negative" as they are easily distinguished. However, some negative samples may hold other textures or part of objects which makes it more difficult to identify. These samples are likely  "Hard Negative".

These "Hard Negative" are difficult to be correctly classified. What we can do about it is to find explicitly those false positive samples during training loops and add them into the training data in order to improve the classifier.


#### 2.1.2 Problems of RCNN
RCNN extracts CNN features for each region proposal by feeding each of them into CNN once at a time, and the proposals selected by Selective Search are approximately 2k for each image, thus this process consumes much time. Adding pre-processing Selective Search, RCNN needs ~47 second per image.

### 2.2 SPP Net (Spatial Pyramid Pooling Network)
To speedup RCNN, SPPNet focuses on how to fix the problem that each proposal is fed into the CNN once a time. The reason behind the problem is the fully connected layers need fixed feature size (i.e., 1 x 21 x 256 in [He et al.,2014](https://arxiv.org/pdf/1406.4729.pdf)) for further classification and regression. Thus SPPNet comes up with an idea that an additional pooling layer called spatial pyramid pooling is inserted right after the last Conv layer and before the Fc layers. The operation of this pooling first projects the region proposals to the Conv features, then divides each feature map (i.e., 60 x 40 x 256 filters) from the last Conv layer into 3 patch scales (i.e., 1,4 and 16 patches, see Fig 4. For example, the patch size is: 60x40 for 1 patch, 30x20 for 4 patches and 15x10 for 16 patches, next operates max pooling on each scaled patch to obtain a 1 x 21(1+4+16) for each feature map, thus we get 1x21x256 fiexd vector for Fc layers.

{{< figure library="true" src="SPPNet_spatial_pyramid_pooling_layer.png" title="Fig 4. The spatial pyramid pooling layer in [SPPNet](https://arxiv.org/pdf/1406.4729.pdf)." lightbox="true" >}}

By proposing spatial pyramid pooling layer, SPPNet is able to reuse the feature maps extracted from CNN by passing the image once through because all information that region proposals need is shared in these feature maps. The only thing we could do next is project the region proposals selected by Selective Search onto these feature maps (**How to project Region Proposals to feature maps? Please go to basic_understanding post for ROI pooling.**). This operation extremely saves time consumption compared to extract feature maps per proposal per forward (like RCNN does). The total speedup of SPPNet is about 100 times compared to RCNN.

## Fast RCNN
{{< figure library="true" src="fast_rcnn2.png" title="Fig 6. The pipeline of Fast RCNN in [this blog](https://towardsdatascience.com/deep-learning-for-object-detection-a-comprehensive-review-73930816d8d9)." lightbox="true" >}}

[Fast RCNN](https://arxiv.org/pdf/1504.08083.pdf) attempts to overcome three notable **drawbacks** of RCNN:
1. **Training a multi-stage pipeline**: fine-tune a ConvNet based on Region Proposals; train SVM classifiers with Conv Features; train bounding-box regressors.
2. **Training is expensive in space and time**: 2.5 GPU-days for 5k images and hundreds of gigabytes of storage.
3. **Speed is slow**: ~47 second per image even on GPU.

**Solutions**:
1. **Combine both classification (replace SVM with softmax) and bounding-box regression into one network with multi-task loss.**
2. **Introduce ROI pooling for: 1. reuse Conv feature maps of one image; 2. speedup both training and testing.** Using VGG16 as backbone network, ROI (Region of Interest) pooling converts all different sizes of region proposals into 7x7x512 feature vector fed into Fc layers. Please go to post **basic_understanding_dl** for more details about ROI pooling.

{{< figure library="true" src="speed_rcnn_fastrcnn.png" title="Fig 6. Speed comparison between RCNN and Fast RCNN in [this blog](https://blog.csdn.net/v_JULY_v/article/details/80170182)." lightbox="true" >}}

### Multi-task Loss for Classification and Bounding-box Regression
$--------------------------------------------------------------------------------------------$

**Symbol Explanation**

$u$       Groundtruth class label, $u \in 0,1,...,K$; To simplify, all background class has $u=0$.

$v$       Groundtruth bounding-box regression target, $v=(v_x,v_y,v_w,v_h)$.

$p$       Descrete probability distribtion (per RoI), $p=(p_0,p_1,...,p_K)$ over $K+1$ categories. $p$ is computed by a softmax over the $K+1$ outputs of a fully connected layer.

$t^u$     Predicted bounding-box vector, $t^u=(t_x^u,t_y^u,t_w^u,t_h^u)$.

$--------------------------------------------------------------------------------------------$

The multi-task loss on each RoI is defined as:

$$L(p,u,t^u,v) = L_{cls}(p,u) + \lambda[u \geqslant 1]L_{loc}(t^u,v)$$
where $L_{cls}(p,u)$=-log$p_u$ is a log loss for groundtruth class $u$. The Iverson bracket indicator function $[u \geqslant 1]$ is 1 when $u \geqslant 1$ (the predicted class is not background), and is 0 otherwise. The $L_{loc}$ term is using smooth L1 loss, which is denoted as:
$$L_{loc}(t^u,v)=\sum_{i \in {x,y,w,h}}smooth_{L_1}(t_i^u-v_i)$$ and
\begin{equation}
smooth_{L_1}(x) = \begin{cases}
                 0.5x^2, &|x| < 1 \newline
                 |x|-0.5,&otherwise
                 \end{cases}
\end{equation}
$smooth_{L_1}(x)$ is a robust $L_1$ loss that is less sensitive to outliers than $L_2$ loss.
## Faster RCNN
{{< figure library="true" src="faster_rcnn.png" title="Fig 7. The pipeline of Faster RCNN in [this blog](https://towardsdatascience.com/deep-learning-for-object-detection-a-comprehensive-review-73930816d8d9)." lightbox="true" >}}

[Faster RCNN](https://arxiv.org/pdf/1506.01497.pdf) focuses on solving the speed bottleneck of Region Proposal Selection as previous RCNN and Fast RCNN separately compute the region proposal by Selective Search on CPU which still consumes much time. To address this problem, a novel subnetwork called RPN (Region Proposal Network) is proposed to combine Region Proposal Selection into ConvNet along with Softmax classifiers and Bounding-box regressors.

{{< figure library="true" src="RPN_mechanism.png" title="Fig 8. The pipeline of RPN in [the paper](https://arxiv.org/pdf/1506.01497.pdf)." lightbox="true" >}}

To adapt the multi-scale scheme of region proposals, the RPN introduces an anchor box. Specifically, RPN has a classifier and a regressor. The classifier is to predict the probability of a proposal holding an object, and the regressor is to correct the proposal coordinates. Anchor is the centre point of the sliding window. For any image, scale and aspect-ratio are two import factors, where scale is the image size and aspect-ratio is width/height. [Ren et al., 2015](https://arxiv.org/pdf/1506.01497.pdf) introduce 9 kinds of anchors, which are scales (1,2,3) and aspect-ratio(1:1,1:2,2:1). Then for the whole image, the number of anchors is $W \times H \times 9$ where $W$ and $H$ are width and height, respectively.

$-------------------------------------------------------------------------------------------$

**Symbol  Explanation**

**$i$**         the index of an anchor in a mini-batch.

**$p_i$**       the probability that the anchor $i$ being an object.

**$p_i^{\*}$**  the groundtruth label $p_i^{\*}$ is 1 if the anchor is positive, and is 0 if the anchor is negative.

**$t_i$**       a vector $(x,y,w,h)$ representing the coordinates of predicted bounding box.

**$t_i^{\*}$**  that of the groundtruth box associated with a positive anchor.

$-------------------------------------------------------------------------------------------$

The RPN also has a multi-task loss just like in Fast RCNN, which is defined as:

$$L({p_i},{t_i}) = \frac{1}{N_{cls}} \sum_i L_{cls}(p_i,p_i^{\*}) + \lambda \frac{1}{N_{reg}} \sum_i p_i^{\*} L_{reg}(t_i, t_i^{\*})$$
where the classification $L_{cls}$ is log loss over two classes (object vs not object). The regression loss $L_{reg}(p_i, p_i^{\*}) = smooth_{L1}(t_i - t_i^{\*})$. The term $p_i^{\*} L_{reg}(t_i, t_i^{\*})$ means the regression loss is activated if $p_i^{\*}=1$ and is disabled if $p_i^{\*}=0$. These two loss term are normalized by $N_{cls}$ and $N_{reg}$ and weighted by a balancing parameter $\lambda$. In implementation, $N_{cls}$ is the number of images in a mini-batch (i.e., $N_{cls}=256$), and the $reg$ term is normalized by the number of anchor locations (i.e., $N_{reg} \sim 2,400$). By default, the $\lambda$ is set to 10.

Therefore, there are four loss functions in one neural network:
1. one is for classifying whether an anchor contains an object or not (anchor good or bad in RPN);
2. one is for proposal bounding box regression (anchor -> groundtruth proposal in RPN);
3. one is for classifying which category that the object belongs to (over all classes in main network);
4. one is for bounding box regression (proposal -> groundtruth bounding-box in main network);

The total speedup comparison between RCNN, Fast RCNN and Faster RCNN is shown below:
{{< figure library="true" src="comparison_speedup_rcnn_fastrcnn_fasterrcnn.png" title="Fig 9. The speedup comparison between RCNN, Fast RCNN and Faster RCNN in [this blog](https://blog.csdn.net/v_JULY_v/article/details/80170182)." lightbox="true" >}}

## Mask RCNN
{{< figure library="true" src="mask_rcnn.png" title="Fig 10. The pipeline of Mask RCNN, which is Faster RCNN + Instance Segmentation + improved RoIAlign Pooling." lightbox="true" >}}

[Mask RCNN](https://arxiv.org/pdf/1703.06870.pdf) has three branches: RPN for region proposal + (a pretrained CNN + Headers for classification and bounding-box regression) + Mask Network for pixel-level instance segmentation. Mask RCNN is developed on Faster RCNN and adds RoIAlign Pooling and instance segmentation to output object masks in a pixel-to-pixel manner. The RoIAlign is proposed to improve RoI for pixel-level segmentation as it requires much more fine-grained alignment than Bounding-boxes. The accurate computation of RoIAlign is described in RoIAlign Pooling for Object Detection in Basic_understanding_dl post.  

{{< figure library="true" src="mask_rcnn_results.png" title="Fig 11. Mask RCNN results on the COCO test set. Image source: [Mask RCNN paper](https://arxiv.org/pdf/1703.06870.pdf)" lightbox="true" >}}

### Mask Loss
During the training, a multi-task loss on each sampled RoI is defined as : $L=L_{cls} + L_{bbox}+L_{mask}$. The $L_{cls}$ and $L_{bbox}$ are identical as those defined in [Faster RCNN](https://arxiv.org/pdf/1506.01497.pdf).

The mask branch has a $K\times m^2$ dimensional output for each RoI, which is $K$ binary masks of resolution $m \times m$, one for each the $K$ classes. Since the mask branch learns a mask for every class with a per-pixel **sigmoid** and a **binary cross-entropy loss**, there is no competition among classes for generating masks. Previous semantic segmentation methods (e.g., [FCN for semantic segmentation](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf)) use a **softmax** and a **multinomial cross-entropy loss**, which causes classification competition among classes.  

$L_{mask}$ is defined as the **average binary mask loss**, which **only includes $k$-th class** if the region is associated with the groundtruth class $k$:
$$L_{mask} = -\frac{1}{m^2} \sum_{1 \leqslant i,j \leqslant m} (y_{ij}log\hat{y}_{ij}^k +(1-y_{ij})log(1-\hat{y}_{ij}^k))$$
where $y_{ij}$ is the label (0 or 1) for a cell $(i,j)$ in the groundtruth mask for the region of size $m \times m$, $\hat{y}_{ij}$ is the predicted value in the same cell in the predicted mask learned by the groundtruth class $k$.

## Summary for R-CNN based Object Detection Methods

{{< figure library="true" src="rcnn-family-summary.png" title="Fig 12. Summary for R-CNN based Object Detection Methods . Image source: [this blog](https://lilianweng.github.io/lil-log/2017/12/31/object-recognition-for-dummies-part-3.html)" lightbox="true" >}}

## 3. Classic One-Stage Methods

### 3.1 YOLO (You Only Look Once)

**Introduction.** YOLO is the first approach removing region proposal and learns an object detector in an end-to-end manner. Due to no region proposal, it frames object detection as a total regression problem which spatially separates bounding boxes and associated class probabilities. The proposed YOLO performs extremely fast (around 45 FPS), but less accuracy than main approaches like Faster RCNN.

#### 3.1.1 Pipeline
{{< figure library="true" src="yolo.png" title="Fig 13. YOLO pipeline. Image source: [original paper](https://arxiv.org/pdf/1506.02640.pdf)" lightbox="true" >}}

1. Resize input image from 224x224 to 448x448;
2. Pre-train a single CNN (DarkNet: 24 conv layer + 2 fc) on ImageNet for classification.
3. Split the input image into $S \times S$ grid, for each cell in the grids:

   3.1. predict coordinates of B boxes, for each box coordiantes: $(x,y,w,h)$ where $x$ and $y$ are the centre location of box, $w$ and $h$ are the width and height of box.
   
   3.2. predict a confidence score, $Pr(obj) \times IoU(trurh, pred)$ where $Pr(obj)$ denote whether the cell contains an object, $Pr(obj)=1$ if it contains an object, otherwise $Pr(obj)=0$. $IoU(truth, pred)$ is the interaction under union.
#### 3.1.2 Loss functions

#### 3.1.3 Difference

#### 3.1.4 Limitations

## Reference
1. https://blog.csdn.net/v_JULY_v/article/details/80170182
2. https://lilianweng.github.io/lil-log/2017/12/31/object-recognition-for-dummies-part-3.html
3. https://medium.com/egen/region-proposal-network-rpn-backbone-of-faster-r-cnn-4a744a38d7f9
4. https://towardsdatascience.com/deep-learning-for-object-detection-a-comprehensive-review-
5. https://zhuanlan.zhihu.com/p/37998710
6. https://blog.csdn.net/heiheiya/article/details/81169758
