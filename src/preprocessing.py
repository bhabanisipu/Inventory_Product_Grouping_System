import pandas as pd
import os


def load_data(file_path):
    """
    Load the inventory sales dataset.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Dataset not found at: {file_path}"
            )

        df = pd.read_csv(file_path)

        if df.empty:
            raise ValueError("The dataset is empty.")

        print("Dataset loaded successfully.")
        return df

    except FileNotFoundError as e:
        print(f"File Error: {e}")
        raise

    except pd.errors.EmptyDataError:
        print("Error: The CSV file is empty.")
        raise

    except pd.errors.ParserError as e:
        print(f"CSV Parsing Error: {e}")
        raise

    except Exception as e:
        print(f"Unexpected error while loading data: {e}")
        raise


def preprocess_data(df):
    """
    Clean and aggregate data by Product ID.
    """

    try:
        if df is None or df.empty:
            raise ValueError("Input DataFrame is empty.")

        required_columns = [
            "Product ID",
            "Category",
            "Inventory Level",
            "Units Sold",
            "Price",
            "Demand"
        ]

        missing_columns = [
            col for col in required_columns
            if col not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}"
            )

        df = df[required_columns].copy()

        numeric_columns = [
            "Inventory Level",
            "Units Sold",
            "Price",
            "Demand"
        ]

        for col in numeric_columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        df.dropna(
            subset=required_columns,
            inplace=True
        )

        df = df[
            df["Product ID"].astype(str).str.strip() != ""
        ]

        if df.empty:
            raise ValueError(
                "No valid rows remain after cleaning."
            )

        product_df = df.groupby("Product ID").agg({
            "Category": "first",
            "Inventory Level": "mean",
            "Units Sold": "sum",
            "Price": "mean",
            "Demand": "mean"
        }).reset_index()

        if product_df.empty:
            raise ValueError(
                "No product records available after aggregation."
            )

        print("Data preprocessing completed successfully.")
        print("Number of products:", len(product_df))

        return product_df

    except ValueError as e:
        print(f"Preprocessing Error: {e}")
        raise

    except Exception as e:
        print(f"Unexpected preprocessing error: {e}")
        raise


def prepare_features(product_df):
    """
    Select numerical features for K-Means clustering.
    """

    try:
        features = [
            "Inventory Level",
            "Units Sold",
            "Price",
            "Demand"
        ]

        if product_df is None or product_df.empty:
            raise ValueError(
                "Product DataFrame is empty."
            )

        missing_columns = [
            col for col in features
            if col not in product_df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing feature columns: {missing_columns}"
            )

        X = product_df[features].copy()

        if X.isnull().any().any():
            raise ValueError(
                "Feature data contains missing values."
            )

        print("Features prepared successfully.")

        return X, features

    except ValueError as e:
        print(f"Feature Preparation Error: {e}")
        raise

    except Exception as e:
        print(f"Unexpected feature preparation error: {e}")
        raise