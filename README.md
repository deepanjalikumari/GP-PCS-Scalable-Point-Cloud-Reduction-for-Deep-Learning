## GP-PCS: Feature-Preserving Point Cloud Simplification with Gaussian Processes
### Project Overview
This project implements GP-PCS (Gaussian Process Point Cloud Simplification) — a state-of-the-art method to drastically reduce the number of points in 3D point clouds while preserving essential geometric features. It leverages Gaussian Processes defined on Riemannian manifolds to adaptively sample points, especially around edges and areas of high curvature, making it highly suitable for downstream 3D shape classification tasks.

The simplified point clouds are used for training and evaluating the deep learning model PointNet on the ModelNet40 dataset, demonstrating efficient 3D shape classification with minimal loss in accuracy.

Features
Feature-Preserving Simplification: Reduces large point clouds by up to 1000x with negligible impact on structure via GP-based surface variation modeling.

Integration with Deep Learning: Simplified point clouds are directly fed into PointNet for 3D object classification.

Efficient and Scalable: Designed to handle large datasets with practical runtime performance and GPU acceleration support.

Empirical Validation: Achieves >90% classification accuracy on ModelNet40 after simplification, with less than 2% drop compared to full-size clouds.

### Installation
Ensure you have Python 3.7+ installed. Clone the repo and install dependencies:

bash
git clone <your-repo-url>
cd <your-repo-directory>
pip install -r requirements.txt
Note: This project requires the following key packages:

numpy

scipy

torch

robust-laplacian

scikit-learn

h5py

### Usage
Prepare the dataset

Download and place the ModelNet40 dataset (HDF5 format with 2048 points per cloud) in the path:

text
/kaggle/input/model40net/modelnet40_hdf5_2048
Run training

Execute the main script or notebook to:

Compute surface variation for each point cloud

Perform GP-PCS-based point cloud simplification

Train PointNet on the simplified dataset

Evaluate

Use the provided evaluation code to test classification accuracy on ModelNet40 test sets.

Configuration
You can modify key hyperparameters in the main script/notebook:

M: Target number of points after simplification (e.g., 1536)

k_init: Initial seed points for Farthest Point Sampling (e.g., 256)

k_add: Points added per iteration during greedy selection (e.g., 256)

GP kernel parameters (sigma_y, kappa, nu) for customized simplification behavior

### Results
Simplification: Point clouds reduced from 2048 to 1536 points, preserving important features.

Classification Accuracy:

Before simplification: ~89.2% (PointNet on full point clouds)

After simplification: ~90.1% (PointNet on GP-PCS simplified clouds)

Runtime: Simplification typically completes in under one minute per large point cloud.

### Pros
High-quality feature preservation in simplification

Minimal impact on classification accuracy

Efficient simplification and scalable pipeline

Easily integrates with existing 3D classification models

### Cons
Hyperparameter tuning may be needed per dataset

Computational overhead due to GP fitting and matrix operations

Very aggressive simplification can impact fine-grained tasks

Requires several external Python libraries and dependencies

### File Structure
modelnet40-gppcs.ipynb — Main Jupyter notebook for data preprocessing, simplification, training, and evaluation.

requirements.txt — Python dependencies file.

README.md — This file.

###Citation
If you use this code or method in your research, please cite the original paper:

GP-PCS: One-shot Feature-Preserving Point Cloud Simplification with Gaussian Processes on Riemannian Manifolds.
