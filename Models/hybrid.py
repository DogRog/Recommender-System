import numpy as np
from scipy.sparse import coo_matrix
import torch
import pickle
from Models.numerical_CBF import NumericalCBF
from Models.collaborative_filtering import ALSRecommender

class HybridRecommender:
    def __init__(
        self,
        alpha=0.5,  # Weight for ALS recommendations
        als_params={
            'factors': 100,
            'regularization': 0.01,
            'alpha': 40,
            'iterations': 15
        },
        cbf_params={
            'batch_size': 1024
        }
    ):
        """
        Initialize hybrid recommender combining ALS and CBF approaches.
        
        Args:
            alpha: Weight for ALS recommendations (1-alpha for CBF)
            als_params: Parameters for ALS model
            cbf_params: Parameters for CBF model
        """
        self.alpha = alpha
        self.als_recommender = ALSRecommender(**als_params)
        self.cbf_recommender = NumericalCBF(**cbf_params)
        
    def fit(self, transactions_df, customers_df, articles_df):
        """
        Fit both recommender models.
        
        Args:
            transactions_df: DataFrame with purchase history
            customers_df: DataFrame with customer information
            articles_df: DataFrame with article information
        """
        # Fit both models
        print("Fitting ALS model...")
        self.als_recommender.fit(transactions_df, customers_df, articles_df)
        
        print("Fitting CBF model...")
        self.cbf_recommender.fit(transactions_df, customers_df, articles_df)
        
        return self
        
    def recommend_items(self, customer_id, n_items=10, filter_already_purchased=True):
        """
        Generate hybrid recommendations.
        
        Args:
            customer_id: ID of the customer
            n_items: Number of recommendations
            filter_already_purchased: Whether to filter purchased items
            
        Returns:
            List of tuples containing (article_id, score)
        """
        try:
            # Get ALS recommendations
            als_recs = self.als_recommender.recommend_items(
                customer_id, 
                n_items=n_items,
                filter_already_purchased=filter_already_purchased
            )
        except ValueError:
            # If customer not in ALS training data, use only CBF
            return self.cbf_recommender.recommend_items(
                customer_id,
                n_items=n_items,
                filter_already_purchased=filter_already_purchased
            )
            
        # Get CBF recommendations
        cbf_recs = self.cbf_recommender.recommend_items(
            customer_id,
            n_items=n_items,
            filter_already_purchased=filter_already_purchased
        )
        
        # Combine recommendations using weighted average
        article_scores = {}
        
        # Normalize ALS scores
        als_scores = np.array([score for _, score in als_recs])
        als_scores = (als_scores - als_scores.min()) / (als_scores.max() - als_scores.min())
        
        # Add weighted ALS scores
        for (article_id, _), norm_score in zip(als_recs, als_scores):
            article_scores[article_id] = self.alpha * norm_score
            
        # Normalize CBF scores
        cbf_scores = np.array([score for _, score in cbf_recs])
        cbf_scores = (cbf_scores - cbf_scores.min()) / (cbf_scores.max() - cbf_scores.min())
        
        # Add weighted CBF scores
        for (article_id, _), norm_score in zip(cbf_recs, cbf_scores):
            if article_id in article_scores:
                article_scores[article_id] += (1 - self.alpha) * norm_score
            else:
                article_scores[article_id] = (1 - self.alpha) * norm_score
                
        # Sort by final scores and return top N
        sorted_recs = sorted(
            article_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:n_items]
        
        return sorted_recs
        
    def save(self, path):
        """Save hybrid model to disk."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
            
    @classmethod
    def load(cls, path):
        """Load hybrid model from disk."""
        with open(path, 'rb') as f:
            return pickle.load(f)
