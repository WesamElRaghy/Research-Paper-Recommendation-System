# Research Paper Recommendation System

This repository implements a Machine Learning-based recommendation system for research papers, leveraging metadata from the arXiv dataset. The project applies text preprocessing, clustering, and dimensionality reduction techniques to build a scalable and effective recommendation system.

## Dataset Overview

Arxiv Dataset https://www.kaggle.com/datasets/Cornell-University/arxiv

The arXiv dataset is a collection of scientific papers that have been submitted to arXiv.org, a repository of electronic preprints of scientific papers in various fields including mathematics, physics, computer science, and more. The arXiv dataset includes papers from a wide range of disciplines and is widely used by researchers to access the latest research in their field. The dataset is maintained by arXiv and is available for download through their website or through third-party services. The dataset can be used for various purposes such as natural language processing, machine learning, and data mining research, and it is a valuable resource for researchers who want to study the latest developments in their fields.

- **ID**: Unique identifier for each paper.
- **Title**: Title of the paper.
- **Abstract**: Summary of the paper's content.
- **Authors**: List of authors.
- **Categories**: Classification of the paper under research domains.
- **Year**: Year of publication.

## Project Workflow
1. **Exploratory Data Analysis (EDA)**:
   - Insights into the distribution of papers by year, abstract length, and categories.
   - Visualization of word clouds for top categories.

2. **Preprocessing**:
   - Text normalization (lowercasing, removing punctuation).
   - Stopword removal and lemmatization.
   - TF-IDF vectorization for feature extraction.

3. **Clustering**:
   - Dimensionality reduction using PCA and UMAP.
   - Clustering with K-Means and Spectral Clustering.
   - Evaluation using Silhouette Score, Davies-Bouldin Score, and Calinski-Harabasz Index.

4. **Recommendation System**:
   - Leveraged cosine similarity on reduced feature embeddings.
   - Implemented K-Nearest Neighbors (KNN) for personalized recommendations.

5. **Visualization**:
   - 2D and 3D embeddings using UMAP and t-SNE.
   - Cluster visualizations with labeled data points.

## Key Results
- **Optimal Clustering**: Achieved using K-Means with 28 clusters.
- **Recommendation Accuracy**: Effective recommendations based on nearest neighbors within clusters.
- **Dimensionality Reduction**: UMAP provided meaningful low-dimensional embeddings for visualization and clustering.

## Requirements
- Python 3.8+
- pandas, numpy, matplotlib, seaborn
- scikit-learn
- umap-learn
- spacy (with `en_core_sci_lg`)
- tensorflow


