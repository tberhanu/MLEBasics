# MLE Basics

A comprehensive collection of machine learning algorithms and techniques implemented from scratch to understand the fundamentals of machine learning engineering.

## Overview

This repository contains implementations of various machine learning algorithms organized by category, along with utilities for data visualization, preprocessing, and model evaluation.

## Directory Structure

```
MLE_Basics/
├── algorithms/
│   ├── supervised/              # Supervised learning algorithms
│   │   ├── boosting_methods/    # Adaptive Boosting, Gradient Boosting
│   │   ├── classification/      # Logistic Regression, Naive Bayes for NLP, etc.
│   │   ├── regression/          # Linear, Polynomial Regression
│   │   └── regression_classification/
│   │       ├── cart/            # Decision Trees, Random Forest
│   │       ├── knn/             # K-Nearest Neighbors
│   │       └── support_vector_machine/  # SVM implementations
│   └── unsupervised/            # Unsupervised learning algorithms
│       ├── centroid_based/      # K-Means
│       ├── density_based/       # DBSCAN
│       ├── dimension_reduction/ # PCA
│       └── hierarchial_clustering/  # Agglomerative, Divisive approaches
├── cross_validation/            # Model validation techniques (Grid Search CV)
├── feature_scaling/             # Normalization, Standardization
├── regularization/              # Regularization techniques
└── visualizations/              # Data visualization and plotting utilities
```

## Algorithms Included

### Supervised Learning

#### Classification
- **Logistic Regression** - Binary and multiclass classification
- **Naive Bayes for NLP** - Probabilistic classifier for natural language processing
- **Categorical Data** - Handling categorical variables

#### Regression
- **Basic Linear Regression** - Simple linear model
- **Linear Regression** - Multiple linear regression
- **Polynomial Regression** - Non-linear regression using polynomial features

#### Tree-Based Methods
- **Decision Trees** - CART algorithm implementation
- **Random Forest** - Ensemble of decision trees

#### Distance-Based Methods
- **K-Nearest Neighbors (KNN)** - Classification using nearest neighbors
- **Support Vector Machine (SVM)** - Linear and non-linear classification
- **SVM Margin Plot** - Visualization of SVM decision boundaries

#### Ensemble Methods
- **Adaptive Boosting** - AdaBoost implementation
- **Gradient Boosting** - Gradient-based boosting algorithm

### Unsupervised Learning

#### Clustering
- **K-Means** - Centroid-based clustering
- **DBSCAN** - Density-based spatial clustering
- **Agglomerative Clustering** - Bottom-up hierarchical clustering
- **Divisive Clustering** - Top-down hierarchical clustering

#### Dimensionality Reduction
- **Principal Component Analysis (PCA)** - Linear dimensionality reduction

### Utilities

#### Feature Scaling
- **Normalization** - Min-Max scaling
- **Standardization** - Z-score normalization

#### Model Validation
- **Grid Search CV** - Hyperparameter tuning with cross-validation

#### Regularization
- **Regularization** - L1 and L2 regularization techniques

#### Visualization Tools
- **Matplotlib Basics** - Basic plotting utilities
- **Seaborn Plots** - Statistical data visualization
- **Pandas Basics** - Data manipulation with Pandas
- **Missing Data** - Handling missing values visualization
- **Outliers** - Outlier detection and visualization
- **Plotting Residuals** - Residual plots for regression analysis
- **Cheatsheet** - Quick reference guide

## Requirements

This project requires Python 3.x with the following libraries:
- NumPy - Numerical computing
- Pandas - Data manipulation
- Scikit-learn - Machine learning utilities
- Matplotlib - Data visualization
- Seaborn - Statistical plotting

## Installation

1. Clone the repository:
```bash
git clone https://github.com/tberhanu/MLEBasics.git
cd MLE_Basics
```

2. Install dependencies:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

## Usage

Each algorithm is implemented as a standalone module that can be imported and used directly:

```python
from algorithms.supervised.classification.logistic_regression import LogisticRegression
from algorithms.unsupervised.centroid_based.k_means import KMeans
from feature_scaling.standardization import StandardScaler
```

Refer to individual files for specific usage examples and documentation.

## Learning Path

### Beginner
1. Start with feature scaling (`feature_scaling/`)
2. Explore basic linear regression (`algorithms/supervised/regression/basic_linear_regression.py`)
3. Learn classification with logistic regression (`algorithms/supervised/classification/logistic_regression.py`)

### Intermediate
1. Study distance-based methods (KNN, SVM)
2. Explore tree-based methods (Decision Trees, Random Forest)
3. Learn ensemble methods (Boosting)

### Advanced
1. Implement unsupervised learning (K-Means, DBSCAN, Hierarchical Clustering)
2. Study dimensionality reduction (PCA)
3. Explore advanced ensemble techniques

## Purpose

This repository is designed for:
- **Learning** - Understanding the fundamentals of machine learning algorithms
- **Reference** - Quick implementation reference for common ML algorithms
- **Practice** - Hands-on experience implementing algorithms from scratch

## Contributing

Feel free to fork, modify, and submit pull requests to improve the implementations or add new algorithms.

## License

This project is open source and available under the MIT License.

## Author

Created by tberhanu

---

**Note:** These implementations are designed for educational purposes. For production use, consider using optimized libraries like Scikit-learn, XGBoost, or TensorFlow.
