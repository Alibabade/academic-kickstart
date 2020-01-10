---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "Training_tricks_dl"
subtitle: ""
summary: "Summary neural network training tricks."
authors: [admin]
tags: [Academic]
categories: [Computer Vision, DL]
date: 2020-01-08T21:03:10Z
lastmod: 2020-01-08T21:03:10Z
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
## Fine-tuning neural networks
In practise, researchers tend to use pre-trained neural networks on datasets like ImageNet to train their own neural network for new tasks due to their dataset perhaps not big enough (compared to millions of images in ImageNet). Thus this type of operation is called fine-tuning the neural network.

There are two typical scenarios:
1. **use the pre-trained CNNs as feature extractors.** For example, we remove the fully connected layers from a pre-trained image classification CNN, then add a classification operator (i.e., softmax and SVM) at the end of left fully convolutional networks to classify images.
2. **fine-tune the pre-trained CNNs.** For example, we preserve part/all of the layers in a pre-trained CNNs, and retrain it on our own dataset. In this case, the front layers extract low-level features which can be used for many tasks (i.e., object recognition/detection and image segmentation), and the rear layers extract high-level features related to specific classification task, thus we only need fine-tune the rear layers.

### How to fine-tune
There are normally four different situations:
1. **New dataset is small and similar to pre-trained dataset.** Since the dataset is small, then retrain the CNN may cause overfitting. And the new dataset is similar to pre-trained dataset, thus we hope the high-level features are similar as well. In this case, we could just use the features extracted from pre-trained CNN and train a classification operator like softmax.
2. **New dataset is small but not similar to pre-trained datasets.** Since the dataset is small then we can not retrain the CNN. And the new dataset is not similar to pre-trained datasets, then we do not use high-level features which means we do not use rear layers. Thus we can just use front layers as feature extractor and training a classification operator like softmax or SVM.
3. **New dataset is big and similar to pre-trained datasets.** We can fine-tune the entire pre-trained CNN.
4. **dataset is big but not similar to pre-trained datasets.** We can fine-tune the entire pre-trained CNN.

In practise, a smaller learning-rate is suggested as the weights in network is already smooth and a larger learning-rate may distort the weights of pre-trained CNN.

### Coding in experiments
In Pytorch, you can set "param.requires_grad = False" to freeze any pre-trained CNN part.
For example, to freeze some layers in BERT model, you could do something like [this](https://github.com/huggingface/transformers/issues/1431):
```python
   if freeze_embeddings:
             for param in list(model.bert.embeddings.parameters()):
                 param.requires_grad = False
             print ("Froze Embedding Layer")

   # freeze_layers is a string "1,2,3" representing layer number
   if freeze_layers is not "":
        layer_indexes = [int(x) for x in freeze_layers.split(",")]
        for layer_idx in layer_indexes:
             for param in list(model.bert.encoder.layer[layer_idx].parameters()):
                 param.requires_grad = False
             print ("Froze Layer: ", layer_idx)
```
