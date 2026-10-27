import os
import joblib

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.preprocessing import (
    load_data,
    preprocess_data,
    prepare_features
)


def train_clustering_model(data_path):
    """
    Preprocess data, train K-Means, and save the model and scaler.
    """

    try:
        df = load_data(data_path)
        product_df = preprocess_data(df)

        X, features = prepare_features(product_df)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train K-Means model
        kmeans = KMeans(
            n_clusters=3,
            init="k-means++",
            random_state=42,
            n_init=10
        )

        kmeans.fit(X_scaled)

        # Predict clusters for existing products
        product_df["Cluster"] = kmeans.predict(X_scaled)

        # Business-friendly cluster labels based on your experiment
        cluster_labels = {
            0: "High-Demand, High-Sales Products",
            1: "Lower-Demand, Higher-Price Products",
            2: "Moderate-Demand Products"
        }

        product_df["Cluster Label"] = (
            product_df["Cluster"].map(cluster_labels)
        )

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models_dir = os.path.join(project_root, "models")
        os.makedirs(models_dir, exist_ok=True)

        # Save model, scaler, and feature names
        joblib.dump(
            kmeans,
            os.path.join(models_dir, "kmeans_model.pkl")
        )

        joblib.dump(
            scaler,
            os.path.join(models_dir, "scaler.pkl")
        )

        joblib.dump(
            features,
            os.path.join(models_dir, "features.pkl")
        )

        print("K-Means model trained and saved successfully.")

        return product_df, kmeans, scaler

    except FileNotFoundError as e:
        print(f"Dataset or file error: {e}")
        raise

    except ValueError as e:
        print(f"Training data error: {e}")
        raise

    except Exception as e:
        print(f"Unexpected error during model training: {e}")
        raise


if __name__ == "__main__":
    try:
        project_root = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )

        data_path = os.path.join(
            project_root,
            "data",
            "sales_data.csv"
        )

        product_df, kmeans, scaler = train_clustering_model(data_path)

        print("\nCluster counts:")
        print(product_df["Cluster Label"].value_counts())

        print("\nSample results:")
        print(
            product_df[
                [
                    "Product ID",
                    "Category",
                    "Cluster",
                    "Cluster Label"
                ]
            ].head()
        )

    except Exception as e:
        print(f"Could not complete clustering: {e}")