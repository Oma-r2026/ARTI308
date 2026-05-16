# Credit Card Customer Segmentation

Unsupervised machine learning project using K-Means clustering to segment credit card customers based on their usage behavior.

## What I Did
- Cleaned data: dropped ID column, filled missing values with mean imputation
- Explored data with histograms, a correlation heatmap, and scatter plots
- Scaled features using StandardScaler (required for distance-based algorithms)
- Used the Elbow Method and Silhouette Score to select K=3
- Visualized clusters using PCA (2D projection)

## Results — 3 Customer Segments

| Cluster | Label | Description |
|---------|-------|-------------|
| 0 | Cash Advance Users | High balance, high cash advance, low purchases |
| 1 | Active Spenders | High purchases, high payments — most valuable segment |
| 2 | Low Activity | Low balance, low spending, low credit limit |
