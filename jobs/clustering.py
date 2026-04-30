import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def load_data():
    data = pd.read_csv("social_ads.csv")
    X = data[["Age", "EstimatedSalary"]]
    return X


def apply_kmeans():

    X = load_data()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(n_clusters=3, random_state=42)
    labels = model.fit_predict(X_scaled)

  
    result = X.copy()
    result["Cluster"] = labels

  
    plt.figure()
    plt.scatter(X["Age"], X["EstimatedSalary"], c=labels)
    plt.xlabel("Age")
    plt.ylabel("Salary")
    plt.title("K-Means Clustering")

    plt.savefig("static/clusters.png")
    plt.close()

    summary = result["Cluster"].value_counts().to_dict()

    return result.to_dict(orient="records"), summary