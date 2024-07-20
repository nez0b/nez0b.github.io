---
layout: distill
title: Quantum Transfer Learning
description: Quantum transfer learning with quantum pooling layer
tags: QML jupyter
categories: quantum-computing
giscus_comments: false
date: 2023-11-20
featured: false
related_posts: true

authors:
  - name: PoJen Wang
    affiliations:
      name: IBM Q-hub, NTU
  - name: Chi Chun Chen
    affiliations:
      name: IBM Q-hub, NTU
  - name: Yuhsuan Tung
    affiliations:
      name: IBM Q-hub, NTU

bibliography: 2018-12-22-distill.bib

# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).
toc:
  - name: Model
    # if a section has subsections, you can add them as follows:
    # subsections:
    #   - name: Example Child Subsection 1
    #   - name: Example Child Subsection 2
  - name: Code

# Below is an example of injecting additional post-specific styles.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---


## Model : Quantum transfer learning with quantum pooling layer
The idea of **transfer leanring**  is to feed the data through pre-trained feature extraction networks first, and train only a small size feed forward netwrok after it to fine tune the moedel with respect to a specific data set.

Since CIFAR100 image data are too large to directly encode in quantum circuits today, we here rely on a "imagenet" pre-trained ResNet18 as feature extraction layer. The image feature is reduced to 4 dimension through this network, and encoded into a 4 qubit circuit network. 

![alt text](/assets/img/qnn_transfer/transfer_learning_general.png)

We then utilize the idea of quantum pooling layer[2] to further reduce the quantum curcuit to a single qubit. A single qubit is sufficient for binary classification by chooing the eigenstate with higher probability.

![alt text](/assets/img/qnn_transfer/transfer_learning_c2qconv.png)
---

## Code

{::nomarkdown}
{% assign jupyter_path = "assets/jupyter/qnn_transfer_learning.ipynb" | relative_url %}
{% capture notebook_exists %}{% file_exists assets/jupyter/qnn_transfer_learning.ipynb %}{% endcapture %}
{% if notebook_exists == "true" %}
{% jupyter_notebook jupyter_path %}
{% else %}

<p>Sorry, the notebook you are looking for does not exist.</p>
{% endif %}
{:/nomarkdown}
