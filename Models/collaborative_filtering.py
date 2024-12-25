import pickle
import implicit
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import svds            

class ALSRecommender:
    """
    A recommendation system using Alternating Least Squares (ALS).
    
    Attributes:
        model: Implicit ALS model instance
        sparse_matrix: Sparse user-item interaction matrix
        customer_index: Mapping of customer IDs to matrix indices
        article_index: Mapping of article IDs to matrix indices
        reverse_customer_index: Reverse mapping of customer indices to IDs
        reverse_article_index: Reverse mapping of article indices to IDs
    """
    
    def __init__(
        self,
        factors=100,
        regularization=0.01,
        alpha=40,
        iterations=15,
        num_threads=4,
        use_gpu=False
    ):
        """
        Initialize the ALS recommender with implicit feedback.
        
        Args:
            factors: Number of latent factors
            regularization: Regularization parameter
            alpha: Confidence scaling parameter
            iterations: Number of ALS iterations
            num_threads: Number of parallel computation threads
            use_gpu: Whether to use GPU acceleration
        """
        self.model = implicit.als.AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            alpha=alpha,
            iterations=iterations,
            num_threads=num_threads,
            use_gpu=use_gpu
        )
        
        # Initialize storage
        self.sparse_matrix = None
        self.customer_index = None
        self.article_index = None
        self.reverse_customer_index = None
        self.reverse_article_index = None
        
    def _create_sparse_matrix_coo(self, transactions_df, customers_df, articles_df):
        """
        Create a sparse purchase matrix using COO format.
        
        Args:
            transactions_df: DataFrame containing purchase transactions
            customers_df: DataFrame containing customer information
            articles_df: DataFrame containing article information
            
        Returns:
            Tuple containing:
                - Sparse matrix in CSR format
                - Dictionary mapping customer IDs to matrix indices
                - Dictionary mapping article IDs to matrix indices
        """
        # Create index mappings
        customer_index = {
            id_: i for i, id_ in enumerate(customers_df['customer_id'])
        }
        article_index = {
            id_: i for i, id_ in enumerate(articles_df['article_id'])
        }
        
        # Vectorized operations for index lookup
        customer_indices = np.array([
            customer_index[cid] for cid in transactions_df['customer_id']
        ])
        article_indices = np.array([
            article_index[aid] for aid in transactions_df['article_id']
        ])
        
        # Create sparse matrix
        sparse_matrix = coo_matrix(
            (
                np.ones(len(transactions_df)),
                (customer_indices, article_indices)
            ),
            shape=(len(customers_df), len(articles_df))
        ).tocsr()
        
        return sparse_matrix, customer_index, article_index
    
    def fit(self, transactions_df, customers_df, articles_df):
        """
        Fit the ALS model using transaction data.
        
        Args:
            transactions_df: DataFrame containing purchase transactions
            customers_df: DataFrame containing customer information
            articles_df: DataFrame containing article information
        """
        # Create sparse matrix and indices
        (
            self.sparse_matrix,
            self.customer_index,
            self.article_index
        ) = self._create_sparse_matrix_coo(
            transactions_df,
            customers_df,
            articles_df
        )
        
        # Create reverse indices
        self.reverse_customer_index = {
            v: k for k, v in self.customer_index.items()
        }
        self.reverse_article_index = {
            v: k for k, v in self.article_index.items()
        }
        
        # Fit the model
        self.model.fit(self.sparse_matrix)

    def recommend_items(self, customer_id, n_items=10, filter_already_purchased=True):
        """
        Get recommendations for a specific customer.
        
        Args:
            customer_id: ID of the customer
            n_items: Number of recommendations to generate
            filter_already_purchased: Whether to exclude purchased items
            
        Returns:
            List of tuples containing (article_id, score)
            
        Raises:
            ValueError: If customer_id is not found in training data
        """
        if customer_id not in self.customer_index:
            raise ValueError(f"Customer ID {customer_id} not found in training data")
            
        # Get recommendations from the model
        user_idx = self.customer_index[customer_id]
        item_scores = self.model.recommend(
            user_idx,
            self.sparse_matrix[user_idx],
            N=n_items,
            filter_already_liked_items=filter_already_purchased
        )
        
        # Convert indices back to article IDs and include scores
        return [
            (self.reverse_article_index[idx], float(score))
            for idx, score in zip(item_scores[0], item_scores[1])
        ]
    
    def save(self, path):
        """
        Save model to disk.
        
        Args:
            path: Path where the model should be saved
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)
            
    @classmethod
    def load(cls, path):
        """
        Load model from disk.
        
        Args:
            path: Path to the saved model file
            
        Returns:
            ALSRecommender: Loaded model instance
        """
        with open(path, 'rb') as f:
            return pickle.load(f)
        

class SVDRecommender:
    """
    A recommendation system using Singular Value Decomposition (SVD).
    
    Attributes:
        factors: Number of latent factors for matrix factorization
        U: User factors matrix
        V: Item factors matrix
        sigma: Singular values
        customer_index: Mapping of customer IDs to matrix indices
        article_index: Mapping of article IDs to matrix indices
        reverse_customer_index: Reverse mapping of customer indices to IDs
        reverse_article_index: Reverse mapping of article indices to IDs
    """
    
    def __init__(self, factors=100):
        """
        Initialize the SVD recommender.
        
        Args:
            factors: Number of latent factors for the SVD decomposition
        """
        self.factors = factors
        
        # Initialize storage for learned matrices
        self.U = None  # User factors
        self.V = None  # Item factors
        self.sigma = None  # Singular values
        
        # Initialize indices
        self.customer_index = None
        self.article_index = None
        self.reverse_customer_index = None
        self.reverse_article_index = None
        self.sparse_matrix = None
        
    def _create_sparse_matrix_coo(self, transactions_df, customers_df, articles_df):
        """
        Create a sparse purchase matrix using COO format.
        
        Args:
            transactions_df: DataFrame containing purchase transactions
            customers_df: DataFrame containing customer information
            articles_df: DataFrame containing article information
            
        Returns:
            Tuple containing:
                - Sparse matrix in CSR format
                - Dictionary mapping customer IDs to matrix indices
                - Dictionary mapping article IDs to matrix indices
        """
        # Create index mappings
        customer_index = {
            id_: i for i, id_ in enumerate(customers_df['customer_id'])
        }
        article_index = {
            id_: i for i, id_ in enumerate(articles_df['article_id'])
        }
        
        # Vectorized operations for index lookup
        customer_indices = np.array([
            customer_index[cid] for cid in transactions_df['customer_id']
        ])
        article_indices = np.array([
            article_index[aid] for aid in transactions_df['article_id']
        ])
        
        # Create sparse matrix
        sparse_matrix = coo_matrix(
            (
                np.ones(len(transactions_df)),
                (customer_indices, article_indices)
            ),
            shape=(len(customers_df), len(articles_df))
        ).tocsr()
        
        return sparse_matrix, customer_index, article_index
        
    def fit(self, transactions_df, customers_df, articles_df):
        """
        Fit the SVD model using transaction data.
        
        Args:
            transactions_df: DataFrame containing purchase transactions
            customers_df: DataFrame containing customer information
            articles_df: DataFrame containing article information
        """
        # Create sparse matrix and indices
        (
            self.sparse_matrix,
            self.customer_index,
            self.article_index
        ) = self._create_sparse_matrix_coo(
            transactions_df,
            customers_df,
            articles_df
        )
        
        # Create reverse indices
        self.reverse_customer_index = {
            v: k for k, v in self.customer_index.items()
        }
        self.reverse_article_index = {
            v: k for k, v in self.article_index.items()
        }
        
        # Perform SVD
        U, sigma, Vt = svds(
            self.sparse_matrix,
            k=min(self.factors, min(self.sparse_matrix.shape) - 1)
        )
        
        # Store decomposed matrices
        self.U = U
        self.sigma = sigma
        self.V = Vt.T
        
    def recommend_items(self, customer_id, n_items=10, filter_already_purchased=True):
        """
        Get recommendations for a specific customer.
        
        Args:
            customer_id: ID of the customer
            n_items: Number of recommendations to generate
            filter_already_purchased: Whether to exclude purchased items
            
        Returns:
            List of tuples containing (article_id, score)
            
        Raises:
            ValueError: If customer_id is not found in training data
        """
        if customer_id not in self.customer_index:
            raise ValueError(f"Customer ID {customer_id} not found in training data")
            
        # Get user vector and calculate scores
        user_idx = self.customer_index[customer_id]
        user_vector = self.U[user_idx]
        scores = user_vector @ np.diag(self.sigma) @ self.V.T
        
        # Filter out purchased items if requested
        if filter_already_purchased:
            user_items = self.sparse_matrix[user_idx].toarray().flatten()
            scores[user_items > 0] = -np.inf
            
        # Get top items
        top_items = np.argsort(scores)[::-1][:n_items]
        
        # Convert indices back to article IDs and include scores
        return [
            (self.reverse_article_index[idx], float(scores[idx]))
            for idx in top_items
        ]

    def save(self, path):
        """
        Save the model to disk.
        
        Args:
            path: Path where the model should be saved
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path):
        """
        Load a saved model from disk.
        
        Args:
            path: Path to the saved model file
            
        Returns:
            SVDRecommender: Loaded model instance
        """
        with open(path, 'rb') as f:
            return pickle.load(f)