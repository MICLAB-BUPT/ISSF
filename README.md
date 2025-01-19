#  ISSF：Weakly-Supervised Temporal Action Localization by Inferring Salient Snippet-Feature

Our paper is available at [[Paper]](https://arxiv.org/pdf/2303.12332)) 

## Abstract
 
>Weakly-supervised temporal action localization aims to locate action regions and identify action categories in untrimmed videos simultaneously by taking only video-level labels as the supervision. Pseudo label generation is a promising strategy to solve the challenging problem, but the current methods ignore the natural temporal structure of the video that can provide rich information to assist such a generation process. In this paper, we propose a novel weakly-supervised temporal action localization method by inferring salient snippet-feature. First, we design a saliency inference module that exploits the variation relationship between temporal neighbor snippets to discover salient snippet-features, which can reflect the significant dynamic change in the video. Secondly, we introduce a boundary refinement module that enhances salient snippet-features through the information interaction unit. Then, a discrimination enhancement module is introduced to enhance the discriminative nature of snippet-features. Finally, we adopt the refined snippet-features to produce high-fidelity pseudo labels, which could be used to supervise the training of the action localization network. Extensive experiments on two publicly available datasets, i.e., THUMOS14 and ActivityNet v1.3, demonstrate our proposed method achieves significant improvements compared to the state-of-the-art methods. 

## Overview

![img](img/model.png)

## Data Preparation
Prepare [THUMOS'14](https://www.crcv.ucf.edu/THUMOS14/) dataset.

## Citation
 
If you find this project useful for your research, please use the following BibTeX entry.

```
@inproceedings{yun2024weakly,
  title={Weakly-Supervised Temporal Action Localization by Inferring Salient Snippet-Feature},
  author={Yun, Wulian and Qi, Mengshi and Wang, Chuanming and Ma, Huadong},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={7},
  pages={6908--6916},
  year={2024}
}

```
