
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


# --------------------------------------------------
# 1. Data Loading Section
# --------------------------------------------------

input_file = r"C:\Users\muham\OneDrive\Desktop\Stage2_2_Output\Stage2_1_Output\Cleaned_Dataset.csv"

output_folder = r"C:\Users\muham\OneDrive\Desktop\Stage2_2_Output"

os.makedirs(output_folder, exist_ok=True)

df = pd.read_csv(input_file)

print("Dataset shape:", df.shape)
print(df.head())


# --------------------------------------------------
# 2. Preparing Data for Clustering
# --------------------------------------------------
# Brief requires segmenting customers by tenure and monthly charges.

if "Churn" not in df.columns:
    raise ValueError("The dataset must contain the Churn column.")

required_cluster_features = ["tenure", "MonthlyCharges"]
missing_features = [
    column for column in required_cluster_features
    if column not in df.columns
]
if missing_features:
    raise ValueError(
        "Missing required clustering features: "
        + ", ".join(missing_features)
    )

# Features used only for K-Means (matches project brief)
cluster_features = required_cluster_features
X = df[cluster_features].copy()

# Extra columns kept for interpretation after clustering
profile_numerical_features = [
    column for column in df.select_dtypes(include=["number"]).columns
    if column not in ["Churn", "Cluster"]
]
profile_categorical_features = [
    column for column in df.columns
    if column not in profile_numerical_features + ["Churn", "Cluster"]
]

print("\nClustering features (brief):")
print(cluster_features)

print("\nProfile numerical features:")
print(profile_numerical_features)

print("\nProfile categorical features:")
print(profile_categorical_features)


# --------------------------------------------------
# 3. Scaling Features
# --------------------------------------------------

preprocessor = ColumnTransformer([
    (
        "numerical",
        StandardScaler(),
        cluster_features
    )
])

X_processed = preprocessor.fit_transform(X)

feature_names = preprocessor.get_feature_names_out()

X_processed = pd.DataFrame(
    X_processed,
    columns=feature_names
)

print("\nProcessed dataset shape:", X_processed.shape)


# --------------------------------------------------
# 4. Applying the Elbow Method
# --------------------------------------------------

k_values = range(1, 11)
inertia_values = []

for k in k_values:
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X_processed)
    inertia_values.append(model.inertia_)

elbow_results = pd.DataFrame({
    "Number_of_Clusters": list(k_values),
    "Inertia": inertia_values
})

print("\nElbow Method Results:")
print(elbow_results.to_string(index=False))

plt.figure(figsize=(8, 5))
plt.plot(list(k_values), inertia_values, marker="o")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method for K-Means Clustering")
plt.xticks(list(k_values))
plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(output_folder, "Elbow_Method.png"),
    dpi=300
)
plt.show()

elbow_results.to_csv(
    os.path.join(output_folder, "Elbow_Results.csv"),
    index=False
)


# --------------------------------------------------
# 5. Comparing Silhouette Scores
# --------------------------------------------------

silhouette_results = []

for k in range(2, 11):
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_processed)

    score = silhouette_score(
        X_processed,
        labels,
        sample_size=min(2000, len(X_processed)),
        random_state=42
    )

    silhouette_results.append({
        "Number_of_Clusters": k,
        "Silhouette_Score": score
    })

silhouette_df = pd.DataFrame(silhouette_results)

print("\nSilhouette Scores:")
print(silhouette_df.to_string(index=False))

silhouette_df.to_csv(
    os.path.join(output_folder, "Silhouette_Results.csv"),
    index=False
)


# --------------------------------------------------
# 6. Selecting the Number of Clusters
# --------------------------------------------------

selected_k = int(
    silhouette_df.loc[
        silhouette_df["Silhouette_Score"].idxmax(),
        "Number_of_Clusters"
    ]
)

print("\nSelected number of clusters:", selected_k)


# --------------------------------------------------
# 7. Applying K-Means Clustering
# --------------------------------------------------

kmeans_model = KMeans(
    n_clusters=selected_k,
    random_state=42,
    n_init=10
)

cluster_labels = kmeans_model.fit_predict(X_processed)

df["Cluster"] = cluster_labels

final_silhouette = silhouette_score(
    X_processed,
    cluster_labels,
    sample_size=min(2000, len(X_processed)),
    random_state=42
)

print("Final silhouette score:", round(final_silhouette, 4))


# --------------------------------------------------
# 8. Calculating Cluster Sizes
# --------------------------------------------------

cluster_sizes = (
    df["Cluster"]
    .value_counts()
    .sort_index()
    .rename_axis("Cluster")
    .reset_index(name="Customer_Count")
)

cluster_sizes["Percentage"] = (
    cluster_sizes["Customer_Count"] / len(df) * 100
).round(2)

print("\nCluster Sizes:")
print(cluster_sizes)

cluster_sizes.to_csv(
    os.path.join(output_folder, "Cluster_Sizes.csv"),
    index=False
)


# --------------------------------------------------
# 9. Analysing Cluster Profiles
# --------------------------------------------------

numerical_profile = df.groupby("Cluster")[
    profile_numerical_features
].mean().round(2)

print("\nNumerical Cluster Profiles:")
print(numerical_profile)

numerical_profile.to_csv(
    os.path.join(output_folder, "Numerical_Cluster_Profiles.csv")
)

categorical_profiles = []

for column in profile_categorical_features:
    profile = pd.crosstab(
        df["Cluster"],
        df[column],
        normalize="index"
    ).mul(100).round(2)

    profile = profile.reset_index()
    profile.insert(1, "Feature", column)

    categorical_profiles.append(profile)

if categorical_profiles:
    categorical_profile_df = pd.concat(
        categorical_profiles,
        ignore_index=True
    )

    categorical_profile_df.to_csv(
        os.path.join(
            output_folder,
            "Categorical_Cluster_Profiles.csv"
        ),
        index=False
    )


# --------------------------------------------------
# 10. Calculating Churn Rate by Cluster
# --------------------------------------------------

churn_values = df["Churn"].astype(str).str.strip().str.lower()

if set(churn_values.unique()).issubset({"0", "1"}):
    df["Churn"] = pd.to_numeric(churn_values, errors="coerce")
else:
    df["Churn"] = churn_values.map({
        "no": 0,
        "yes": 1
    })

churn_summary = df.groupby("Cluster").agg(
    Customer_Count=("Churn", "count"),
    Churn_Rate=("Churn", "mean")
)

churn_summary["Churn_Rate"] = (
    churn_summary["Churn_Rate"] * 100
).round(2)

print("\nChurn Rate by Cluster:")
print(churn_summary)

churn_summary.to_csv(
    os.path.join(output_folder, "Churn_Rate_By_Cluster.csv")
)


# --------------------------------------------------
# 11. Visualising Cluster Sizes
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.bar(
    cluster_sizes["Cluster"].astype(str),
    cluster_sizes["Customer_Count"]
)

plt.xlabel("Cluster")
plt.ylabel("Number of Customers")
plt.title("Customer Distribution Across Clusters")
plt.tight_layout()

plt.savefig(
    os.path.join(output_folder, "Cluster_Sizes.png"),
    dpi=300
)
plt.show()


# --------------------------------------------------
# 12. Visualising Clusters Using PCA
# --------------------------------------------------

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_processed)

pca_df = pd.DataFrame({
    "Component_1": X_pca[:, 0],
    "Component_2": X_pca[:, 1],
    "Cluster": cluster_labels
})

print(
    "\nVariance explained by two components:",
    round(pca.explained_variance_ratio_.sum() * 100, 2),
    "%"
)

plt.figure(figsize=(8, 6))

for cluster in sorted(pca_df["Cluster"].unique()):
    cluster_data = pca_df[pca_df["Cluster"] == cluster]

    plt.scatter(
        cluster_data["Component_1"],
        cluster_data["Component_2"],
        label=f"Cluster {cluster}",
        alpha=0.6
    )

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title(f"K-Means Customer Clusters (K = {selected_k})")
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(output_folder, "PCA_Clusters.png"),
    dpi=300
)
plt.show()

pca_df.to_csv(
    os.path.join(output_folder, "PCA_Results.csv"),
    index=False
)


# --------------------------------------------------
# 13. Saving the Final Results
# --------------------------------------------------

df.to_csv(
    os.path.join(output_folder, "Clustered_Dataset.csv"),
    index=False
)

X_processed.to_csv(
    os.path.join(output_folder, "Processed_Clustering_Data.csv"),
    index=False
)

joblib.dump(
    preprocessor,
    os.path.join(output_folder, "Clustering_Preprocessor.pkl")
)

joblib.dump(
    kmeans_model,
    os.path.join(output_folder, "KMeans_Model.pkl")
)

print("\nClustering analysis completed successfully.")
print("Selected clusters:", selected_k)
print("Output folder:", output_folder)