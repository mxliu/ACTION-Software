[//]: # (References)
[//]: # (\bibliography{path/to/yout/references})

<font color=blue size=5>
** News: Source code and pretrained models of three new graph convolutional networks (HGCN, HKGCN, and HKGAT) defined in hyperbolic space have been added! March 05, 2025
</font>

# ACTION-Software-for-Functional-MRI-Analysis

The Augmentation and Computation Toolbox for braIn netwOrk aNalysis (ACTION) is an open-source Python software, designed for functional MRI data augmentation, brain network construction and visualization, extraction of brain network features, and intelligent analysis of brain networks based on AI models pretrained on 3,800+ resting-state fMRI scans. 
Through a graphics user interface, the ACTION aims to provide users with comprehensive, convenient, and easy-to-use fMRI data analysis services, helping users simplify the processing pipeline and improve work efficiency.

Software Download: https://mingxia.web.unc.edu/action/

# 1. FMRI Data Augmentation
## 1.1. BOLD Signal Augmentation 

1. Upsampling
2. Downsampling
3. Slicing
4. Noise Jittering
5. Self-Defined Algorithm
 
## 1.2. Graph Augmentation

1. Random Node Dropping
2. Hub-Preserving Node Dropping
3. Random Edge Perturbation
4. Weight-Dependent Edge Removal
5. Subgraph Cropping
6. Attribute Masking
7. Self-Defined Algorithm

# 2. Brain Network Construction

1. Pearson’s Correlation (PC)
2. Mutual Information (MI)
3. Partial Correlation
4. Spearman’s Correlation (SC)
5. High-Order Functional Connectivity (HOFC)
6. Sparse Representation (SR)
7. Low-rank Representation (LR)
8. Self-Defined Algorithm

# 3. Brain Network Feature Extraction

1. Node-level network feature extraction
2. Graph-level network feature extraction
   
# 4. Artificial Intelligence Model Construction

## 4.1. Machine Learning Model Construction 
  
### Machine Learning Model

1. Support Vector Mahince (SVM)
2. Support Vector Regression (SVR)  
3. Random Forest (RF) 
4. Extreme Gradient Boosting (XGBoost)  
5. K-Nearest Neighbors (KNN)
6. Self-Defined Algorithm

### Data Partition
1. K-fold Cross Validation
2. Random Partition

### Feature Selection
1. Principle Component Analysis (PCA)
2. Univariate Feature Selection (UFS)
3. T-test

## 4.2. Deep Learning Foundation Model Construction

### Overview

This module focuses on (1) pretraining foundation models that can easily adapt to downstream tasks for fMRI analysis, and (2) implementing popular federated learning methods to facilitate multi-site fMRI studies. 
1. The feature encoders of ten deep learning models are pretrained on 3,806 unlabeled fMRI scans from public cohorts in a self-supervised learning manner. In each directory, the `Pretrain_*.py` and `Finetune_*.py` scripts are the main functions for pretraining and fine-tuning a specific encoder, respectively.
2. The source code of five federated learning methods are released to the public, where the users can use the pretrained backbone in this toolbox as the encoder for federated learning. 

### Pretrained Foundation Model

In this project, we pretrain several popular graph convolutional networks on auxiliary fMRI scans. The pretrained models include:

1. Graph Convolutional Network (GCN)
2. Graph Attention Network (GAT)
3. Graph Isomorphism Network (GIN)
4. BrainGNN
5. BrainNetCNN
6. Spatio-Temporal Attention Graph Isomorphism Network (STAGIN)
7. Spatio-Temporal Graph Convolutional Network (STGCN)
8. Graph SAmple and aggreGatE (GraphSAGE) 
9. Transformer
10. Modularity-constrained Graph Neural Network (MGNN)
11. Hyperbolic Graph Convolutional Network (HGCN)
12. Hyperbolic Kernel Graph Convolutional Network (HKGCN)
13. Hyperbolic Kernel Graph Attention Network (HKGAT) 

### Federated Learning Strategy

To facilitate multi-site fMRI research, this toolbox includes several popular federated learning algorithms:
1. Federated Averaging (FedAvg)
2. Federated Proximal (FedProx)
3. Model-contrastive federated learning (MOON)
4. Personalized Federated learning with Moreau envelope (pFedMe)
5. Local Global Federated Averaging (LGFedAvg)

### Usage

1. Please fine-tune the pretrained encoders for various fMRI-based analyses.
Note that the `Finetune_*.py` script is just an example of how to finetune our pretrained encoders for classification tasks, and one can modify this code according to different downstream tasks.

- For 'GCN', 'GIN', 'GAT', 'BrainNetCNN', 'GraphSAGE', 'STAGIN', 'STGCN', 'MGNN', and 'Transformer', 
please input your to-be-analyzed data in the `Data` class of the corresponding `Finetune_*.py`,
including fMRI time series with shape of (nsub,nlength,nroi) and label with the shape of (nsub,).

- For 'BrainGNN', please input your to-be-analyzed data in `BrainGNN_data.py`,
including fMRI time series with shape of (nsub,nlength,nroi) and label with the shape of (nsub,).

2. For model fine-tunning, please ensure that your data includes fMRI time series with a shape of (nsub, nlength, nroi) and labels with a shape of (nsub,), where:
- `nsub`: the number of subjects
- `nlength`: the length of fMRI time series
- `nroi`: the number of regions-of-interest (ROIs)
 
3. A default GCN model is used in each federated learning method.
Users can also employ other backbone models for federated learning. 

4. Acknowledgments to the following public projects:
[SimSiam](https://github.com/facebookresearch/simsiam),
[UCGL](https://github.com/mxliu/Unsupervised-Contrastive-Graph-Learning),
[GCN](https://github.com/tkipf/gcn),
[GAT](https://github.com/gordicaleksa/pytorch-GAT),
[Transformer](https://github.com/gordicaleksa/pytorch-original-transformer/tree/main),
[BrainGNN](https://github.com/xxlya/BrainGNN_Pytorch),
[BrainNetCNN](https://github.com/nicofarr/brainnetcnnVis_pytorch/tree/master),
[GraphSAGE](https://github.com/williamleif/graphsage-simple),
[STAGIN](https://github.com/egyptdj/stagin),
[STGCN](https://github.com/sgadgil6/cnslab_fmri),
[MGNN](https://github.com/mxliu/Brain-Modularity-Prior-Modeling-for-fMRI-Analysis);
[FedAvg](https://github.com/shaoxiongji/federated-learning/tree/master),
[FedProx](https://github.com/mmendiet/FedAlign/blob/main/methods/fedprox.py),
[MOON](https://github.com/QinbinLi/MOON),
[pFedMe](https://github.com/CharlieDinh/pFedMe/tree/master),
[LGFedAvg](https://github.com/pliang279/LG-FedAvg).

# Contact
If you have any problem with our code or have some suggestions, please feel free to contact us: 

- Yuqi Fang (yuqi9199@gmail.com)
- Junhao Zhang (zhangjunhao123333@163.com)
- Linmin Wang (w17860361619@163.com)
- Qianqian Wang (qqw@email.unc.edu)
- Mingxia Liu (mingxia_liu@med.unc.edu)


# Citation
If you use this toolbox in your research, please cite this paper:

```
@article{action2025,
  author = {Fang, Yuqi and Zhang, Junhao and Wang, Linmin and Wang, Qianqian and Liu, Mingxia},  
  title = {ACTION: Augmentation and computation toolbox for brain network analysis with functional MRI},  
  journal = {NeuroImage},
  year = {2025}
}
```


# License
The University of North Carolina at Chapel Hill (UNC-CH) holds all rights to the ACTION software, which is available at no cost for users in academia. 
Individuals are permitted to distribute and modify ACTION under the conditions of the GNU General Public License issued by the Free Software Foundation. 
Commercial or industrial entities interested in utilizing ACTION must reach out to both UNC-CH and the tool's author for permission.  

