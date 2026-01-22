# PCA in Supervised and Unsupervised Workflows

## PCA as an Unsupervised Technique
- PCA doesn’t use labels  
- It analyzes only the structure of the input features  
- Its objective is to find directions of maximum variance  

## PCA in Supervised Learning Pipelines
- Preprocessing for classification (e.g., SVM, logistic regression)  
- Preprocessing for regression  
- Noise reduction before training a supervised model  
- Feature engineering to reduce dimensionality and reduce overfitting  

---

# Understanding PCA Through the Matrix Perspective

## Data Matrix as Rows and Columns
- Think of your dataset like a **database table**  
  - **Rows** = individual samples  
  - **Columns** = features (variables)  
- PCA is primarily interested in the **variance of each feature (column)** and how features vary together  
- The covariance matrix summarizes these relationships and becomes the foundation for PCA  

---

# Key Mathematical Concepts in PCA

## Eigenvalues and Eigenvectors
- PCA is built on the eigen-decomposition of the covariance matrix  
- **Eigenvectors** represent the principal components (directions of maximum variance)  
- **Eigenvalues** measure how much variance each principal component captures  
- Larger eigenvalues correspond to more informative components  
- Sorting eigenvalues in descending order ranks the principal components  

## Covariance Matrix
- PCA computes the covariance matrix of the standardized data  
- This matrix captures how features (columns) vary with one another  
- Eigen-decomposition of this matrix yields the principal components  

## Dimensionality Reduction
- PCA projects data onto the top \(k\) eigenvectors  
- This reduces dimensionality while preserving most of the variance  
- Components with small eigenvalues are often discarded  

## Orthogonality
- Principal components are **orthogonal** (uncorrelated)  
- This eliminates redundancy and simplifies downstream models  

## Reconstruction
- PCA allows approximate reconstruction of the original data using only the top components  
- Reconstruction error increases as fewer components are kept  

---

# Intuition Behind PCA
- PCA rotates the coordinate system to align with directions of maximum variance  
- It finds the “best-fitting” lower-dimensional subspace  
- It reveals structure in high-dimensional data that may not be obvious in the original feature space
