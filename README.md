# BABS
The official PyTorch implementation of our  IEEE GRSL paper (Background-Aware Band Selection for Object Tracking in Hyperspectral Videos (BABS))
# Abstract
Hyperspectral images (HSIs) contain many bands that can be used to obtain object material information for object tracking and remote sensing. Nevertheless, neighboring bands of HSIs are often highly correlated, and a large number of bands increase the complexity of model learning. This issue is worsened by the shortage of labeled hyperspectral videos (HSVs) for fine-tuning pretrained deep neural networks. To tackle these challenges, this letter introduces a novel background-aware band selection method to model spatial changes of an object and its corresponding local region, which is capable of selecting discriminative bands for object representation while reducing computational complexity. Specifically, the object and local regions of each band are compared with other bands to obtain their dissimilarity scores. Guided by these scores, the top three bands are selected and form a three-channel image. This image is then fed into an object tracker. Experimental results demonstrate the efficiency and effectiveness of the proposed method on a benchmark hyperspectral object-tracking dataset.

# Results
|                | AUC    | DP    | link           |
|----------------|------- |-------|----------------|
| HOT 2022       | 0.616  | 0.901 | [Example](#)   |
| HOT 2023       | 0.515  | 0.770 | [Link](#)      |
| HOT 2024       | 0.381  | 0.576 | [Another](#)   |

# If this work is helpful to you, please cite it as:
@article{islam2023background,
  title={Background-aware band selection for object tracking in hyperspectral videos},
  author={Islam, Mohammad Aminul and Zhou, Jun and Zhang, Weichuan and Gao, Yongsheng},
  journal={IEEE Geoscience and Remote Sensing Letters},
  year={2023},
  publisher={IEEE}
}
# Some of the other papers you might find useful:

