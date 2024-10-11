# ML_Project-Customer_Segmentation_Using_K_Means 🛍️📊

This project explores the task of segmenting customers into different groups based on their purchasing behavior using **K-Means Clustering**. By analyzing various customer attributes, the model identifies distinct clusters, allowing businesses to tailor their marketing strategies.

## Data
This directory contains the dataset (`Mall_Customers.csv`) used for the project. The dataset includes the following features:

- **Customer ID**: Unique identifier for each customer.
- **Gender**: Gender of the customer (male or female).
- **Age**: Age of the customer.
- **Annual Income**: Customer's yearly income (in thousands).
- **Spending Score**: A score (1-100) indicating how much the customer spends and their general purchasing behavior.

> **Note:** You may need to adjust the dataset features based on your specific project requirements.

## Notebooks
This directory contains the Jupyter Notebook (`Customer_Segmentation_using_K_Means_Clustering.ipynb`) that guides you through the entire process of data exploration, preprocessing, clustering using K-Means, and visualization.

## Running the Project
The Jupyter Notebook walks through the following steps:

### Data Loading and Exploration:
- Load the dataset and explore the basic statistics.
- Visualize relationships between features and clusters.

### Data Preprocessing:
- Handle missing values (if any).
- Scale numerical features like `Annual Income` and `Spending Score`.
- Encode categorical variables (e.g., `Gender`).

### Determining the Optimal Number of Clusters:
- Use the **Elbow Method** to determine the optimal number of clusters.
  
```python
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
```

### Model Training with K-Means Clustering:
- Train the K-Means model to group the customers into clusters.
- Visualize clusters using scatter plots.

### Model Evaluation:
- Evaluate the model using within-cluster sum of squares (WCSS) and visual inspection of cluster separation.

### Visualization of Results:
- Visualize clusters based on features like **Annual Income** and **Spending Score** to identify distinct customer segments.

## Customization
Modify the Jupyter Notebook to:
- Experiment with different feature engineering techniques.
- Try other clustering algorithms (e.g., **Hierarchical Clustering**, **DBSCAN**) for comparison.
- Explore advanced techniques like customer segmentation based on **unsupervised neural networks**.

## Resources
- Sklearn K-Means Documentation: [https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
- Kaggle Mall Customers Dataset: [https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python](https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial-in-python)

## Further Contributions
Extend this project by:
- Incorporating additional customer data (e.g., transaction history, online activity) for more granular segmentation.
- Applying clustering to other domains like market basket analysis or product recommendation.
- Using clustering results to drive business decisions for targeted marketing campaigns.

By leveraging **K-Means Clustering** and customer segmentation techniques, businesses can gain deeper insights into their customer base and enhance their marketing and service delivery. This project provides a foundation for further exploration in customer segmentation and analysis.

