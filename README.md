# Customer Churn Analysis

Team project for telecom customer churn analysis.

## Repository Structure

### Stage 1 - Data Engineering
Folder: `Stage1_Data_Engineering`

- Cleaned and preprocessed the customer dataset
- Handled missing values, encoded categorical variables, scaled numeric features
- Created train/test splits for predictive modelling
- Saved a reusable preprocessing pipeline

**Key outputs**
- `Cleaned_Dataset.csv`
- `Complete_Preprocessed_Data.csv`
- `Preprocessed_Training_Set.csv` / `Preprocessed_Testing_Set.csv`
- `Preprocessing_Pipeline.pkl`

### Stage 2 - Clustering Analysis
Folder: `Stage2_Clustering`

- Segmented customers using K-Means on **tenure** and **MonthlyCharges**
- Selected optimal clusters with Elbow method + Silhouette score (`K = 4`)
- Profiled clusters and calculated churn rate by segment
- Visualised results with Elbow, cluster-size, and PCA charts

**Key finding**
- Cluster 3: short tenure + high monthly charges + ~49% churn (retention priority)

**Key outputs**
- `Clustering_Analysis.py`
- `Clustered_Dataset.csv`
- Elbow / Silhouette / PCA results and charts
- Cluster profiles and churn-by-cluster summaries

## How to run Stage 2
1. Open `Stage2_Clustering/Clustering_Analysis.py` in Thonny
2. Ensure the Stage 1 cleaned dataset path is correct
3. Install packages: pandas, scikit-learn, matplotlib, joblib
4. Run the script (F5)
