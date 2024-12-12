"""
Recommender System Evaluation Module

This module provides functionality for evaluating various recommender system models
using metrics such as Recall@K, catalog coverage, and recommendation diversity.
"""

import os
import time
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

# Local imports
from Models.numerical_CBF import NumericalCBF
from Models.hybrid import HybridRecommender
from Models.collaborative_filtering import ALSRecommender, SVDRecommender


def load_and_verify_data():
    """
    Load and verify the data files exist and have the expected structure.
    """
    try:
        # Load data files
        data_dir = "Data"
        customers_path = os.path.join(data_dir, "customers.pkl")
        articles_path = os.path.join(data_dir, "articles.pkl")
        transactions_path = os.path.join(data_dir, "transactions.pkl")

        if not os.path.exists(customers_path):
            raise FileNotFoundError(f"customers.pkl not found in {data_dir} directory")
        if not os.path.exists(articles_path):
            raise FileNotFoundError(f"articles.pkl not found in {data_dir} directory")
        if not os.path.exists(transactions_path):
            raise FileNotFoundError(f"transactions.pkl not found in {data_dir} directory")

        customers = pd.read_pickle(customers_path)
        articles = pd.read_pickle(articles_path)
        transactions = pd.read_pickle(transactions_path)

        # Print data info
        print("\nData Overview:")
        print("\nCustomers DataFrame:")
        print(customers.info())
        print("\nArticles DataFrame:")
        print(articles.info())
        print("\nTransactions DataFrame:")
        print(transactions.info())

        # Check required columns
        required_columns = {
            'customers': ['customer_id'],
            'articles': ['article_id'],
            'transactions': ['customer_id', 'article_id', 't_dat']
        }

        for df_name, cols in required_columns.items():
            df = locals()[df_name]
            missing_cols = [col for col in cols if col not in df.columns]
            if missing_cols:
                raise KeyError(f"Missing required columns in {df_name}: {missing_cols}")

        return customers, articles, transactions

    except FileNotFoundError as e:
        print(f"Error loading data: {str(e)}")
        raise
    except KeyError as e:
        print(f"Error verifying data: {str(e)}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        raise


class RecommenderEvaluator:
    """Class for evaluating recommender system models."""

    @staticmethod
    def _calculate_metric_at_k(actual, predicted, k, metric_type='precision'):
        """
        Calculate Mean Precision@K or Mean Recall@K.

        Args:
            actual: Dictionary of actual items purchased by each user.
            predicted: List of recommended items for each user.
            k: Number of items to consider.
            metric_type: 'precision' or 'recall'.

        Returns:
            float: Mean Precision@K or Mean Recall@K.
        """
        metrics = []
        predicted_k = [prediction[:k] for prediction in predicted]
        predicted_sets = [set(prediction) for prediction in predicted_k]

        for user_id, predicted_set in zip(actual.keys(), predicted_sets):
            actual_items = set(actual[user_id])
            if not predicted_set:
                continue

            n_relevant_and_rec = len(actual_items & predicted_set)

            if metric_type == 'precision':
                metric = n_relevant_and_rec / k
            elif metric_type == 'recall':
                metric = n_relevant_and_rec / len(actual_items)
            else:
                raise ValueError("metric_type must be 'precision' or 'recall'")
            metrics.append(metric)

        return np.mean(metrics) if metrics else 0.0

    @staticmethod
    def calculate_mean_precision_at_k(actual, candidates, k):
        """Calculate Mean Precision@K."""
        return RecommenderEvaluator._calculate_metric_at_k(actual, candidates, k, metric_type='precision')

    @staticmethod
    def calculate_mean_recall_at_k(actual, predicted, k):
        """Calculate Mean Recall@K."""
        return RecommenderEvaluator._calculate_metric_at_k(actual, predicted, k, metric_type='recall')

    @staticmethod
    def calculate_coverage(all_candidates, catalog_items):
        """Calculate catalog coverage of recommendations."""
        predicted_items = set().union(*map(set, all_candidates))
        return len(predicted_items) / len(catalog_items)

    @staticmethod
    def calculate_diversity(all_candidates, item_features):
        """Calculate recommendation diversity using item features."""
        if not item_features:
            return 0.0

        def calculate_pair_diversity(item1, item2):
            if item1 not in item_features or item2 not in item_features:
                return 0.0

            vec1, vec2 = item_features[item1], item_features[item2]
            similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            return 1 - similarity

        diversities = []
        for candidates in all_candidates:
            pair_diversities = []
            for i, item1 in enumerate(candidates):
                for item2 in candidates[i + 1:]:
                    pair_diversities.append(calculate_pair_diversity(item1, item2))

            if pair_diversities:
                diversities.append(np.mean(pair_diversities))

        return np.mean(diversities) if diversities else 0.0

    @staticmethod
    def evaluate_model(model, users, items, purchases, k=10, item_features=None):
        """Evaluate a recommendation model using multiple metrics."""
        start_time = time.time()
        all_candidates = []

        print("Evaluating model recommendations...")
        for user in tqdm(users):
            candidates = model.recommend_items(user, n_items=k)
            predicted = [c[0] for c in candidates]
            all_candidates.append(predicted)

        elapsed_time = time.time() - start_time

        metrics = {
            f'Precision@{k}': RecommenderEvaluator.calculate_mean_precision_at_k(purchases, all_candidates, k),
            f'Recall@{k}': RecommenderEvaluator.calculate_mean_recall_at_k(purchases, all_candidates, k),
            'Catalog Coverage': RecommenderEvaluator.calculate_coverage(all_candidates, set(items.keys())),
            'Elapsed Time': elapsed_time
        }

        if item_features is not None:
            metrics['Diversity'] = RecommenderEvaluator.calculate_diversity(
                all_candidates, item_features
            )

        print(f"Evaluated model in {elapsed_time:.2f} seconds")
        for metric, value in metrics.items():
            if metric != 'Elapsed Time':
                print(f"{metric}: {value:.4f}")

        return metrics


class ModelManager:
    """Class for managing and evaluating multiple recommender models."""

    def __init__(self):
        self.models = {}

    def initialize_models(self, train_data, customers, articles):
        """Initialize all recommender models."""
        print("\nInitializing models...")
        try:
            # Initialize NumericalCBF
            print("Initializing NumericalCBF...")
            self.models['NumericalCBF'] = NumericalCBF(
                output_dir='recommendations',
                device='mps',
                batch_size=256
            )
            print("Training data shape:", train_data.shape)
            print("Articles data shape:", articles.shape)
            print("Customers data shape:", customers.shape)

            # Debug column names
            print("\nArticles columns:", articles.columns.tolist())
            print("Training data columns:", train_data.columns.tolist())

            self.models['NumericalCBF'].fit(train_data, customers, articles)

            # Initialize HybridRecommender
            print("Initializing HybridRecommender...")
            self.models['HybridRecommender'] = HybridRecommender(
                alpha=0.5,
                als_params={
                    'factors': 100,
                    'regularization': 0.01,
                    'alpha': 40,
                    'iterations': 15
                },
                cbf_params={'batch_size': 256}
            )
            self.models['HybridRecommender'].fit(train_data, customers, articles)

            # Initialize other recommenders
            self._initialize_recommender(train_data, customers, articles, 'ALSRecommender')
            self._initialize_recommender(train_data, customers, articles, 'SVDRecommender')

        except Exception as e:
            print(f"Error initializing models: {str(e)}")
            raise

    def _initialize_recommender(self, train_data, customers, articles, model_type):
        """Initialize a recommender model."""
        print(f"Initializing {model_type}...")
        weights_path = f"Weights/{model_type.lower()}.pkl"
        if os.path.exists(weights_path):
            if model_type == 'ALSRecommender':
                self.models[model_type] = ALSRecommender.load(weights_path)
            elif model_type == 'SVDRecommender':
                self.models[model_type] = SVDRecommender.load(weights_path)
        else:
            if model_type == 'ALSRecommender':
                self.models[model_type] = ALSRecommender(
                    factors=100,
                    regularization=0.01,
                    alpha=40,
                    iterations=15,
                    num_threads=8,
                    use_gpu=False
                )
            elif model_type == 'SVDRecommender':
                self.models[model_type] = SVDRecommender(factors=100)
            
            self.models[model_type].fit(train_data, customers, articles)
            self.models[model_type].save(weights_path)

    def _create_item_features(self, articles_df, feature_columns=None):
        """
        Create item features dictionary for diversity calculation.

        Args:
            articles_df: DataFrame containing article information.
            feature_columns: List of columns to use for feature creation.

        Returns:
            Dictionary mapping article_ids to their feature vectors.
        """
        if feature_columns is None:
            feature_columns = [
                'product_type_no',
                'garment_group_no',
                'colour_group_code',
                'section_no',
                'perceived_colour_value_id',
                'perceived_colour_master_id'
            ]

        # Normalize features
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(articles_df[feature_columns])

        # Create dictionary mapping article_ids to feature vectors
        item_features = {}
        for idx, article_id in enumerate(articles_df['article_id']):
            item_features[article_id] = features_normalized[idx]

        return item_features

    def evaluate_all_models(self, users, items, purchases, k=100, save_path=None, articles_df=None):
        """
        Evaluate all initialized models.

        Args:
            users: List of user IDs.
            items: Dictionary of item information.
            purchases: Dictionary of user purchase history.
            k: Number of recommendations to evaluate.
            save_path: Optional path to save results as JSON.
            articles_df: Optional DataFrame containing article features for diversity calculation.
        """
        all_metrics = {}
        for model_name, model in self.models.items():
            print(f"\nEvaluating {model_name}...")
            # Create item features for diversity if articles_df is provided
            item_features = None
            if articles_df is not None:
                item_features = self._create_item_features(articles_df)

            metrics = RecommenderEvaluator.evaluate_model(
                model=model,
                users=users,
                items=items,
                purchases=purchases,
                k=k,
                item_features=item_features
            )
            all_metrics[model_name] = metrics

        results_df = pd.DataFrame(all_metrics).T

        if save_path:
            # Convert DataFrame to JSON-serializable format
            results_dict = {}
            for model in results_df.index:
                results_dict[model] = {}
                for metric in results_df.columns:
                    value = results_df.loc[model, metric]
                    if isinstance(value, (np.int64, np.int32, np.int16, np.int8)):
                        value = int(value)
                    elif isinstance(value, (np.float64, np.float32, np.float16)):
                        value = float(value)
                    results_dict[model][metric] = value

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump(results_dict, f, indent=4)
            print(f"\nResults saved to: {save_path}")

        return results_df


def main():
    """Main function to run the evaluation pipeline and evaluate model performance."""
    try:
        # Load and verify data
        print("Loading and verifying data...")
        customers, articles, transactions = load_and_verify_data()

        # Ensure all IDs are strings
        articles['article_id'] = articles['article_id'].astype(str)
        customers['customer_id'] = customers['customer_id'].astype(str)
        transactions['article_id'] = transactions['article_id'].astype(str)
        transactions['customer_id'] = transactions['customer_id'].astype(str)

        # Prepare data
        print("\nPreparing data...")
        transactions["t_dat"] = pd.to_datetime(transactions["t_dat"])
        split_date = pd.to_datetime('2020-09-16')

        train = transactions[transactions["t_dat"] < split_date].copy()
        val = transactions[transactions["t_dat"] >= split_date].copy()

        # Process purchase histories
        train_purchases = train.groupby('customer_id')['article_id'].agg(list).reset_index()
        val_purchases = val.groupby('customer_id')['article_id'].agg(list).reset_index()

        # Get common users
        common_users = set(train_purchases['customer_id']) & set(val_purchases['customer_id'])
        print(f"Number of common users: {len(common_users)}")

        # Filter data
        train_filtered = train_purchases[train_purchases['customer_id'].isin(common_users)]
        val_filtered = val_purchases[val_purchases['customer_id'].isin(common_users)]

        # Initialize and evaluate models
        model_manager = ModelManager()
        model_manager.initialize_models(train, customers, articles)

        # Define save path for results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_path = f"Results/evaluation_results_{timestamp}.json"

        # Create item features for diversity calculation
        print("\nCreating item features for diversity evaluation...")
        item_features = model_manager._create_item_features(articles)

        results_df = model_manager.evaluate_all_models(
            users=list(common_users),
            items=dict(articles['article_id']),
            purchases=val_filtered.set_index('customer_id')['article_id'].to_dict(),
            k=100,
            save_path=results_path,
            articles_df=articles  # Pass articles DataFrame for diversity calculation
        )

        print("\nFinal Results:")
        print(results_df)

    except Exception as e:
        print(f"\nError in main execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()
