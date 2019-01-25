# 3D-brain segmentation using deep neural network and Gaussian mixture model

## Introduction
- Automatically segmenting important brain tissues such as gray matter (GM), white matter (WM), and cerebrospinal fluid (CSF) from high-quality magnetic resonance images (MRI) has played a crucial role in clinical diagnostics and neuroscience research for helping to assess many diseases.
- We propose a novel method utilizing Gaussian Mixture Model (GMM), Convoluion neural network (CNN) and Deep Neural Network (DNN) to classify each voxel of 3D MRI brain images.
- The empirical results on the dataset IBSR 18 [4] show that our proposed method outperforms 13 states-of-the-art algorithms, surpassing all the other methods by a significant margin. 
![Optional Text](../master/illustrations/Picture1.png)

## Method
### System overview
- We divide voxels into two groups of certain (easy-to-classify) voxels and uncertain (hard-to-classify) voxels using the uncertain-voxel detector.
- The certain voxels are classified using the GMM. 
- The uncertain voxels are classified using a DNN.
![Optional Text](../master/illustrations/Picture2.png)

### Gaussian Mixture Model for certain voxels
- The histogram of intensity of voxels of each brain region (as shown in Figure 3) has the shape of a normal distribution and peaks of these three histograms are obviously seperated.
- Using GMM to capture the shapes of intensity distribution of three brain regions’ voxels. The parameters are optimized by the Expectation Maximization algorithm.
- Applying the trained model to classify new voxels.

### Uncertain-voxel detector
- A combination of CNN and DNN is used to classify a voxel to be certain/uncertain voxel based on the information including intensity and coordinates of itself and its surrounding voxels in the grayscale MRI.
- “Certain”/“Uncertain” label is assigned to voxels that are correctly/incorrectly predicted by the GMM.  
- Training dataset is created by applying the GMM on training data to determine voxels predicted correctly /incorrectly.
- The architecture of this combination can be described in Figure 4. The input features are the intensity of surrounding voxels (along 3 axes) and normalized coordinates of the predicted voxels. The output layer has the size of two, indicating the prediction of certain/uncertain voxels. 
- Usually, a testing brain is predicted to has approximately 78% of certain voxels and 22% of uncertain voxels.
![Optional Text](../master/illustrations/Picture3.png)

### Deep Neural Network for uncertain voxels
- Another combination of CNN and DNN is used to classify a voxel to be in GM/WM/CSF class, also based on the imformation including intensity and coordinates of itself and its surrounding voxels in the grayscale MRI.
- The corresponding architecture is as same as the uncertain-voxel detector’s. However, the output layer has the size of three (for GM/WM/CSF prediction).
![Optional Text](../master/illustrations/Picture4.png)

## Original Paper
Duy M. H. Nguyen, Huy T. Vu, Huy Q. Ung, Binh T. Nguyen. "3D-brain segmentation using deep neural network and Gaussian mixture model",  2017 IEEE Winter Conference on Applications of Computer Vision (WACV).
https://ieeexplore.ieee.org/document/7926679/
