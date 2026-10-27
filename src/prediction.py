import os
import joblib
import pandas as pd


def load_saved_objects():
    """
    Load the trained K-Means model, scaler, and feature names.
    """

    try:
        project_root = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )

        models_dir = os.path.join(project_root, "models")

        model_path = os.path.join(
            models_dir, "kmeans_model.pkl"
        )
        scaler_path = os.path.join(
            models_dir, "scaler.pkl"
        )
        features_path = os.path.join(
            models_dir, "features.pkl"
        )

        for path in [model_path, scaler_path, features_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Required saved file not found: {path}. "
                    "Train the model first."
                )

        kmeans = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        features = joblib.load(features_path)

        return kmeans, scaler, features

    except FileNotFoundError as e:
        print(f"Model File Error: {e}")
        raise

    except Exception as e:
        print(f"Error loading saved objects: {e}")
        raise


def predict_product_group(inventory_level, units_sold, price, demand):
    """
    Predict the cluster and business group for a product.
    """

    try:
        kmeans, scaler, features = load_saved_objects()

        # Prepare the input in the same order as training features
        input_data = pd.DataFrame([{
            "Inventory Level": inventory_level,
            "Units Sold": units_sold,
            "Price": price,
            "Demand": demand
        }])

        input_data = input_data[features]

        # Validate numeric input
        if input_data.isnull().any().any():
            raise ValueError("All feature values must be provided.")

        if not all(
            pd.api.types.is_numeric_dtype(input_data[col])
            for col in features
        ):
            raise ValueError("Feature values must be numeric.")

        if not all(
            input_data[col].map(
                lambda value: float("-inf") < value < float("inf")
            ).all()
            for col in features
        ):
            raise ValueError("Feature values must be finite numbers.")

        if (input_data < 0).any().any():
            raise ValueError("Feature values cannot be negative.")

        # Scale input with the previously fitted scaler
        scaled_input = scaler.transform(input_data)

        # Predict cluster
        cluster = int(kmeans.predict(scaled_input)[0])

        # Map cluster to business label
        cluster_labels = {
            0: "High-Demand, High-Sales Products",
            1: "Lower-Demand, Higher-Price Products",
            2: "Moderate-Demand Products"
        }

        label = cluster_labels.get(
            cluster,
            "Unclassified Product Group"
        )

        return {
            "cluster": cluster,
            "product_group": label
        }

    except ValueError as e:
        print(f"Input or Prediction Error: {e}")
        raise

    except FileNotFoundError:
        raise

    except Exception as e:
        print(f"Unexpected prediction error: {e}")
        raise


if __name__ == "__main__":
    try:
        result = predict_product_group(
            inventory_level=300,
            units_sold=340000,
            price=60,
            demand=105
        )

        print("\nPrediction Result")
        print("-----------------")
        print("Cluster:", result["cluster"])
        print("Product Group:", result["product_group"])

    except Exception as e:
        print(f"Prediction could not be completed: {e}")