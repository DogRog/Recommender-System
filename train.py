import os
import time
import json
import argparse
import yaml
import pandas as pd
from dataclasses import dataclass
from abc import ABC, abstractmethod

from Models.collaborative_filtering import ALSRecommender, SVDRecommender
from Models.content_based_filtering import ContentBasedFiltering, NumericalCBF
from Models.hybrid import HybridRecommender


@dataclass
class DataPaths:
    '''Data paths configuration'''
    data_dir: str = "Data"
    customers_path: str = "customers.pkl"
    articles_path: str = "articles.pkl"
    transactions_path: str = "transactions.pkl"

    def get_full_path(self, filename: str) -> str:
        return os.path.join(self.data_dir, filename)

    def verify_paths(self) -> None:
        for path in [self.customers_path, self.articles_path, self.transactions_path]:
            full_path = self.get_full_path(path)
            if not os.path.exists(full_path):
                raise FileNotFoundError(f"{path} not found in {self.data_dir} directory")


class DataLoader:
    '''Handles loading and preprocessing of data'''
    
    def __init__(self, data_paths: DataPaths):
        self.data_paths = data_paths

    def load_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load all data files"""
        try:
            self.data_paths.verify_paths()
            
            customers = pd.read_pickle(self.data_paths.get_full_path(self.data_paths.customers_path))
            articles = pd.read_pickle(self.data_paths.get_full_path(self.data_paths.articles_path))
            transactions = pd.read_pickle(self.data_paths.get_full_path(self.data_paths.transactions_path))

            # Convert IDs to strings
            for df in [articles, customers, transactions]:
                for col in ['article_id', 'customer_id']:
                    if col in df.columns:
                        df[col] = df[col].astype(str)

            return customers, articles, transactions

        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise


class ModelFactory:
    '''Creates recommender models based on configuration'''
    
    @staticmethod
    def create_model(config):
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
        elif model_type == 'CBF':
            return ContentBasedFiltering(max_text_features=config['max_text_features'])
        elif model_type == 'NumericalCBF':
            return NumericalCBF()
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


class ModelTrainer:
    '''Handles model training and saving'''
    
    def __init__(self, data_loader: DataLoader, model_factory: ModelFactory, save_dir: str = "Models"):
        self.data_loader = data_loader
        self.model_factory = model_factory
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def train_model(self, config):
        '''Train model with given configuration'''
        try:
            # Load data
            print("Loading data...")
            customers, articles, transactions = self.data_loader.load_data()
            
            # Create and train model
            print(f"Training {config['model_type']}...")
            start_time = time.time()
            model = self.model_factory.create_model(config)
            model.fit(transactions, customers, articles)
            training_time = time.time() - start_time
            
            # Save model and config
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(self.save_dir, f"{config['model_type']}_{timestamp}.pkl")
            config_path = os.path.join(self.save_dir, f"{config['model_type']}_{timestamp}_config.json")
            
            model.save(model_path)
            with open(config_path, 'w') as f:
                json.dump({
                    'config': config,
                    'training_time': training_time,
                    'timestamp': timestamp
                }, f, indent=4)
            
            print(f"\nTraining completed in {training_time:.2f} seconds")
            print(f"Model saved to: {model_path}")
            print(f"Config saved to: {config_path}")
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            raise


def main():
    parser = argparse.ArgumentParser(description='Train recommender model')
    parser.add_argument('--config', type=str, required=True, 
                       help='Path to configuration file')
    parser.add_argument('--save-dir', type=str, default='Weights',
                       help='Directory to save trained model')
    args = parser.parse_args()
    
    try:
        # Load configuration
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Initialize components
        data_paths = DataPaths()
        data_loader = DataLoader(data_paths)
        model_factory = ModelFactory()
        model_trainer = ModelTrainer(data_loader, model_factory, args.save_dir)
        
        # Train model
        model_trainer.train_model(config)

    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()