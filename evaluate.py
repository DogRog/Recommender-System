"""
Recommender System Evaluation Module

This module provides functionality for evaluating various recommender system models
using metrics such as Recall@K, catalog coverage, and recommendation diversity.
"""

import os
import time
from typing import Dict, List, Optional, Set, Tuple, Union, Any
import json
import argparse
import yaml
import wandb
import numpy as np
import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from abc import ABC, abstractmethod

from Models.collaborative_filtering import ALSRecommender, SVDRecommender
from Models.content_based_filtering import ContentBasedFiltering, NumericalCBF
from Models.hybrid import HybridRecommender


@dataclass
class DataPaths:
    """Data paths configuration"""
    data_dir: str = "Data"
    customers_path: str = "customers.pkl"
    articles_path: str = "articles.pkl"
    transactions_path: str = "transactions.pkl"

    def get_full_path(self, filename: str) -> str:
        """Get full path for a given filename"""
        return os.path.join(self.data_dir, filename)

    def verify_paths(self) -> None:
        """Verify all data files exist"""
        for path in [self.customers_path, self.articles_path, self.transactions_path]:
            full_path = self.get_full_path(path)
            if not os.path.exists(full_path):
                raise FileNotFoundError(f"{path} not found in {self.data_dir} directory")


class DataLoader:
    """Handles loading and verification of data"""
    
    REQUIRED_COLUMNS = {
        'customers': ['customer_id'],
        'articles': ['article_id'],
        'transactions': ['customer_id', 'article_id', 't_dat']
    }

    def __init__(self, data_paths: DataPaths):
        self.data_paths = data_paths

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load and verify all data files"""
        try:
            self.data_paths.verify_paths()
            
            customers = pd.read_pickle(self.data_paths.get_full_path(self.data_paths.customers_path))
            articles = pd.read_pickle(self.data_paths.get_full_path(self.data_paths.articles_path))
            transactions = pd.read_pickle(self.data_paths.get_full_path(self.data_paths.transactions_path))

            self._print_data_info(customers, articles, transactions)
            self._verify_columns(customers, articles, transactions)
            
            return customers, articles, transactions

        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise

    def _print_data_info(self, customers: pd.DataFrame, articles: pd.DataFrame, 
                        transactions: pd.DataFrame) -> None:
        """Print information about loaded dataframes"""
        print("\nData Overview:")
        for name, df in [("Customers", customers), ("Articles", articles), 
                        ("Transactions", transactions)]:
            print(f"\n{name} DataFrame:")
            print(df.info())

    def _verify_columns(self, customers: pd.DataFrame, articles: pd.DataFrame, 
                       transactions: pd.DataFrame) -> None:
        """Verify required columns exist in dataframes"""
        dfs = {'customers': customers, 'articles': articles, 'transactions': transactions}
        
        for df_name, cols in self.REQUIRED_COLUMNS.items():
            df = dfs[df_name]
            missing_cols = [col for col in cols if col not in df.columns]
            if missing_cols:
                raise KeyError(f"Missing required columns in {df_name}: {missing_cols}")


class MetricCalculator:
    """Calculates various recommendation metrics"""
    
    @staticmethod
    def calculate_metric_at_k(actual: Dict[str, List[str]], predicted: List[List[str]], 
                            k: int, metric_type: str = 'precision') -> float:
        """Calculate Mean Precision@K or Mean Recall@K"""
        if metric_type not in ['precision', 'recall']:
            raise ValueError("metric_type must be 'precision' or 'recall'")
            
        metrics = []
        predicted_k = [prediction[:k] for prediction in predicted]
        predicted_sets = [set(prediction) for prediction in predicted_k]

        for user_id, predicted_set in zip(actual.keys(), predicted_sets):
            if not predicted_set:
                continue

            actual_items = set(actual[user_id])
            n_relevant_and_rec = len(actual_items & predicted_set)

            metric = (n_relevant_and_rec / k if metric_type == 'precision' 
                     else n_relevant_and_rec / len(actual_items))
            metrics.append(metric)

        return np.mean(metrics) if metrics else 0.0

    @staticmethod
    def calculate_coverage(all_candidates: List[List[str]], 
                         catalog_items: Set[str]) -> float:
        """Calculate catalog coverage of recommendations"""
        predicted_items = set().union(*map(set, all_candidates))
        return len(predicted_items) / len(catalog_items)

    @staticmethod
    def calculate_diversity(all_candidates: List[List[str]], 
                          item_features: Dict[str, np.ndarray],
                          use_batched: bool = False,
                          batch_size: int = 1000) -> float:
        """Calculate recommendation diversity using item features"""
        if not item_features:
            return 0.0
            
        if use_batched:
            return MetricCalculator._calculate_diversity_batched(
                all_candidates, item_features, batch_size)
        else:
            return MetricCalculator._calculate_diversity_vectorized(
                all_candidates, item_features)
    
    @staticmethod
    def _calculate_diversity_vectorized(all_candidates: List[List[str]], 
                                      item_features: Dict[str, np.ndarray]) -> float:
        """Vectorized diversity calculation without batching"""
        
        def calculate_list_diversity(candidates: List[str]) -> float:
            # Filter valid items and get their feature vectors
            valid_items = [item for item in candidates if item in item_features]
            if len(valid_items) < 2:
                return 0.0
                
            # Stack vectors into a matrix and normalize
            vectors = np.vstack([item_features[item] for item in valid_items])
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors_normalized = vectors / norms
            
            # Calculate all pairwise similarities at once using matrix multiplication
            similarities = np.dot(vectors_normalized, vectors_normalized.T)
            
            # Extract upper triangle (excluding diagonal) using efficient indexing
            n = len(vectors)
            indices = np.triu_indices(n, k=1)
            upper_similarities = similarities[indices]
            
            # Convert similarities to diversities
            diversities = 1 - upper_similarities
            
            return np.mean(diversities) if len(diversities) > 0 else 0.0
        
        # Calculate diversity for each recommendation list
        diversities = [calculate_list_diversity(candidates) 
                      for candidates in all_candidates 
                      if len(candidates) > 1]
        
        return np.mean(diversities) if diversities else 0.0

    @staticmethod
    def _calculate_diversity_batched(all_candidates: List[List[str]], 
                                   item_features: Dict[str, np.ndarray],
                                   batch_size: int) -> float:
        """Batched diversity calculation for large datasets"""
        total_diversity = 0.0
        total_batches = 0
        
        # Process recommendations in batches
        for i in range(0, len(all_candidates), batch_size):
            batch = all_candidates[i:i + batch_size]
            if not batch:
                continue
                
            # Calculate diversity for current batch
            batch_diversities = []
            for candidates in batch:
                valid_items = [item for item in candidates if item in item_features]
                if len(valid_items) < 2:
                    continue
                    
                vectors = np.vstack([item_features[item] for item in valid_items])
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                vectors_normalized = vectors / norms
                
                similarities = np.dot(vectors_normalized, vectors_normalized.T)
                n = len(vectors)
                indices = np.triu_indices(n, k=1)
                upper_similarities = similarities[indices]
                
                diversities = 1 - upper_similarities
                if len(diversities) > 0:
                    batch_diversities.append(np.mean(diversities))
            
            if batch_diversities:
                total_diversity += sum(batch_diversities)
                total_batches += len(batch_diversities)
        
        return total_diversity / total_batches if total_batches > 0 else 0.0


class RecommenderEvaluator:
    """Evaluates recommender system models"""

    def __init__(self, metric_calculator: MetricCalculator):
        self.metric_calculator = metric_calculator

    def evaluate_model(self, model: 'BaseRecommender', 
                      users: List[str], items: Dict[str, str], purchases: Dict[str, List[str]], 
                      k: int = 100, item_features: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, float]:
        """Evaluate a recommendation model using multiple metrics"""
        start_time = time.time()
        
        print("Evaluating model recommendations...")
        all_candidates = []
        for user in tqdm(users):
            candidates = model.recommend_items(user, n_items=k)
            predicted = [c[0] for c in candidates]
            all_candidates.append(predicted)

        elapsed_time = time.time() - start_time

        metrics = {
            f'Precision@{k}': self.metric_calculator.calculate_metric_at_k(
                purchases, all_candidates, k, 'precision'),
            f'Recall@{k}': self.metric_calculator.calculate_metric_at_k(
                purchases, all_candidates, k, 'recall'),
            'Catalog Coverage': self.metric_calculator.calculate_coverage(
                all_candidates, set(items.keys())),
            'Elapsed Time': elapsed_time
        }

        if item_features is not None:
            metrics['Diversity'] = self.metric_calculator.calculate_diversity(
                                                                all_candidates, 
                                                                item_features,
                                                                use_batched=True,
                                                                batch_size=1000
                                                            )

        self._print_metrics(metrics)
        return metrics

    def _print_metrics(self, metrics: Dict[str, float]) -> None:
        """Print evaluation metrics"""
        print(f"Evaluated model in {metrics['Elapsed Time']:.2f} seconds")
        for metric, value in metrics.items():
            if metric != 'Elapsed Time':
                print(f"{metric}: {value:.4f}")


class BaseRecommender(ABC):
    """Abstract base class for recommender models"""
    
    @abstractmethod
    def fit(self, train_data: pd.DataFrame, customers: pd.DataFrame, articles: pd.DataFrame) -> None:
        """Train the recommender model"""
        pass

    @abstractmethod
    def recommend_items(self, user_id: str, n_items: int = 100) -> List[Tuple[str, float]]:
        """Generate recommendations for a user"""
        pass


class ModelFactory:
    """Factory for creating recommender models"""
    
    @staticmethod
    def create_model(config: Dict) -> 'BaseRecommender':
        """Create and return a recommender model based on configuration"""
        model_type = config['model_type']
        
        if model_type == 'ALSRecommender':
            return ALSRecommender(
                factors=config['factors'],
                regularization=config['regularization'],
                alpha=config['alpha'],
                iterations=config['iterations']
            )
        elif model_type == 'SVDRecommender':
            return SVDRecommender(factors=config['factors'])
        elif model_type == 'NumericalCBF':
            return NumericalCBF()
        elif model_type == 'CBF':
            return ContentBasedFiltering(max_text_features=config['max_text_features'])
        elif model_type == 'HybridRecommender':
            return HybridRecommender(
                alpha=config['alpha_weight'],
                als_params={
                    'factors': config['factors'],
                    'regularization': config['regularization'],
                    'alpha': config['alpha'],
                    'iterations': config['iterations']
                }
            )
        else:
            raise ValueError(f"Invalid model type: {model_type}")


class FeatureProcessor:
    """Processes and creates item features"""
    
    DEFAULT_FEATURE_COLUMNS = [
        'product_type_no',
        'garment_group_no',
        'colour_group_code',
        'section_no',
        'perceived_colour_value_id',
        'perceived_colour_master_id'
    ]

    @staticmethod
    def create_item_features(articles_df: pd.DataFrame, 
                           feature_columns: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """Create normalized item features dictionary"""
        feature_columns = feature_columns or FeatureProcessor.DEFAULT_FEATURE_COLUMNS
        
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(articles_df[feature_columns])
        
        return dict(zip(articles_df['article_id'], features_normalized))


class ResultsManager:
    """Handles saving and loading of evaluation results"""
    
    def __init__(self, results_dir: str = "Results"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
    def save_results(self, 
                    results: Dict[str, float], 
                    model_config: Dict[str, Any], 
                    timestamp: Optional[str] = None) -> str:
        """Save evaluation results and model configuration to a JSON file"""
        if timestamp is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            
        # Combine results and configuration
        save_data = {
            "timestamp": timestamp,
            "metrics": results,
            "config": model_config,
            "wandb_run_id": wandb.run.id if wandb.run is not None else None
        }
        
        # Create filename using model type and timestamp
        model_type = model_config.get("model_type", "unknown")
        filename = f"{model_type}_{timestamp}.json"
        filepath = os.path.join(self.results_dir, filename)
        
        # Save to JSON
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=4)
            
        print(f"\nResults saved to: {filepath}")
        return filepath
        
    def load_results(self, filepath: str) -> Dict[str, Any]:
        """Load results from a JSON file"""
        with open(filepath, 'r') as f:
            return json.load(f)
            
    def get_results_summary(self) -> pd.DataFrame:
        """Create a summary DataFrame of all results"""
        all_results = []
        
        for filename in os.listdir(self.results_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.results_dir, filename)
                result_data = self.load_results(filepath)
                
                # Flatten the nested structure
                summary = {
                    'timestamp': result_data['timestamp'],
                    'model_type': result_data['config']['model_type'],
                    'wandb_run_id': result_data['wandb_run_id']
                }
                
                # Add metrics
                summary.update(result_data['metrics'])
                
                # Add key configuration parameters
                for key, value in result_data['config'].items():
                    if key != 'model_type':  # Already included
                        summary[f'config_{key}'] = value
                        
                all_results.append(summary)
                
        return pd.DataFrame(all_results)


class ExperimentManager:
    """Manages the execution of recommendation system experiments"""
    
    def __init__(self, data_loader: DataLoader, model_factory: ModelFactory, 
                 evaluator: RecommenderEvaluator, results_manager: ResultsManager):
        self.data_loader = data_loader
        self.model_factory = model_factory
        self.evaluator = evaluator
        self.results_manager = results_manager

    def prepare_data(self, transactions: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Set[str]]:
        """Prepare training and validation data using last 8 weeks (7 for training, 1 for testing)"""
        transactions["t_dat"] = pd.to_datetime(transactions["t_dat"])
        
        # Get the last date in the dataset
        last_date = transactions["t_dat"].max()
        
        # Calculate split dates using weeks
        total_period = 8  # Total weeks to consider
        test_period = 1   # Weeks for testing
        train_period = 7  # Weeks for training
        
        split_start = last_date - pd.Timedelta(weeks=total_period)  # Start of training period
        split_date = last_date - pd.Timedelta(weeks=test_period)    # Start of test period
        
        # Filter to last 8 weeks and split into train/val
        recent_transactions = transactions[transactions["t_dat"] >= split_start].copy()
        train = recent_transactions[recent_transactions["t_dat"] < split_date].copy()
        val = recent_transactions[recent_transactions["t_dat"] >= split_date].copy()
        
        print(f"\nData split information:")
        print(f"Training period: {split_start.date()} to {split_date.date()} ({train_period} weeks)")
        print(f"Testing period: {split_date.date()} to {last_date.date()} ({test_period} week)")
        print(f"Training set size: {len(train):,} transactions")
        print(f"Test set size: {len(val):,} transactions")

        # Keep track of individual transactions for training
        train_data = train.copy()
        
        # For validation, we want the grouped format
        val_purchases = val.groupby('customer_id')['article_id'].agg(list).reset_index()

        # Find common users between train and val
        train_users = set(train_data['customer_id'].unique())
        val_users = set(val_purchases['customer_id'])
        common_users = train_users & val_users
        print(f"Number of common users: {len(common_users):,}")
        print(f"Percentage of val users in train: {(len(common_users) / len(val_users)) * 100:.2f}%")

        return (train_data[train_data['customer_id'].isin(common_users)],
                val_purchases[val_purchases['customer_id'].isin(common_users)],
                common_users)

    def run_experiment(self, config: Dict) -> None:
        """Run a single experiment with given configuration"""
        try:
            customers, articles, transactions = self.data_loader.load_data()
            
            # Ensure IDs are strings
            for df in [articles, customers, transactions]:
                for col in ['article_id', 'customer_id']:
                    if col in df.columns:
                        df[col] = df[col].astype(str)

            train_filtered, val_filtered, common_users = self.prepare_data(transactions)
            
            model = self.model_factory.create_model(config)
            model.fit(train_filtered, customers, articles)
            
            item_features = FeatureProcessor.create_item_features(articles)
            
            results = self.evaluator.evaluate_model(
                model=model,
                users=list(common_users),
                items=dict(zip(articles['article_id'], articles['article_id'])),
                purchases=val_filtered.set_index('customer_id')['article_id'].to_dict(),
                k=100,
                item_features=item_features
            )
            
            # Save results
            self.results_manager.save_results(results, config)
            
            # Log to wandb if available
            if wandb.run is not None:
                wandb.log(results)
            
        except Exception as e:
            print(f"Error during experiment: {str(e)}")
            import traceback
            print(f"Traceback:\n{traceback.format_exc()}")
            raise


def main():
    """Main function to run the evaluation pipeline with configurable run mode"""
    parser = argparse.ArgumentParser(description='Run model evaluation')
    parser.add_argument('--config', type=str, required=True, 
                       help='Path to configuration file')
    parser.add_argument('--results-dir', type=str, default='Results',
                       help='Directory to save evaluation results')
    args = parser.parse_args()
    
    try:
        # Load configuration
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Initialize components
        data_paths = DataPaths()
        data_loader = DataLoader(data_paths)
        metric_calculator = MetricCalculator()
        evaluator = RecommenderEvaluator(metric_calculator)
        model_factory = ModelFactory()
        results_manager = ResultsManager(args.results_dir)
        experiment_manager = ExperimentManager(data_loader, model_factory, evaluator, results_manager)
        
        # Check run mode from config
        run_mode = config.get('run_mode', 'sweep')  # Default to sweep for backward compatibility
        
        if run_mode == 'single':
            # Run single evaluation
            wandb.login()
            with wandb.init(project=config['project_name']) as run:
                experiment_manager.run_experiment(config)
        else:  # run_mode == 'sweep'
            # Set up wandb sweep for grid search
            sweep_config = {
                'method': 'grid',
                'metric': {
                    'name': 'Precision@100',
                    'goal': 'maximize'
                },
                'parameters': {
                    param: values for param, values in config.items() 
                    if isinstance(values, dict) and 'values' in values
                }
            }
            
            def run_sweep():
                """Execute a single sweep run"""
                with wandb.init() as run:
                    # Combine fixed and sweep parameters
                    run_config = {k: v for k, v in config.items() 
                                if not isinstance(v, dict)}
                    run_config.update(wandb.config)
                    
                    experiment_manager.run_experiment(run_config)
            
            # Initialize wandb
            wandb.login()
            
            # Start sweep
            sweep_id = wandb.sweep(
                sweep_config,
                project=config['project_name']
            )
            
            # Run sweep
            wandb.agent(sweep_id, function=run_sweep)

    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()