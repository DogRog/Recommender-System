import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class GPUCBRecommender:
    """
    A PyTorch-accelerated smart recommendation system for H&M products.
    Supports CUDA (NVIDIA) and MPS (Apple Silicon) acceleration.
    """
    
    def __init__(self, articles_df, transactions_df, customers_df, device=None):
        """
        Initialize the recommender system with necessary data.

        Args:
            articles_df (pd.DataFrame): Product catalog
            transactions_df (pd.DataFrame): Transaction history
            customers_df (pd.DataFrame): Customer information
            device (str, optional): Specify 'cuda', 'mps', or 'cpu'. If None, best available device is selected.
        """
        self.articles_df = articles_df.copy()
        self.transactions_df = transactions_df.copy()
        self.customers_df = customers_df.copy()
        self.tfidf = TfidfVectorizer(stop_words='english')
        
        # Device selection logic
        self.device = self._get_device(device)
        print(f"Using device: {self.device}")
        
        # Initialize feature matrix
        self.feature_matrix = None
        self.feature_matrix_torch = None
        
    def _get_device(self, device):
        """
        Determine the appropriate device for computation.
        
        Args:
            device (str): Requested device ('cuda', 'mps', 'cpu', or None)
            
        Returns:
            torch.device: Selected computing device
        """
        if device is None:
            # Auto-select best available device
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                return torch.device('cpu')
        
        # User-specified device
        if device == 'cuda' and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. Falling back to CPU.")
            return torch.device('cpu')
        elif device == 'mps' and (not hasattr(torch.backends, 'mps') or not torch.backends.mps.is_available()):
            print("Warning: MPS requested but not available. Falling back to CPU.")
            return torch.device('cpu')
        
        return torch.device(device)
    
    def _to_torch(self, data):
        """
        Convert data to PyTorch tensor and move to appropriate device.
        
        Args:
            data: NumPy array or scipy sparse matrix
            
        Returns:
            torch.Tensor on specified device
        """
        if isinstance(data, csr_matrix):
            # Convert CSR matrix to COO format for PyTorch
            coo = data.tocoo()
            values = torch.FloatTensor(coo.data)
            indices = torch.LongTensor(np.vstack((coo.row, coo.col)))
            tensor = torch.sparse_coo_tensor(
                indices, values, torch.Size(data.shape)
            ).to_dense()
        else:
            tensor = torch.FloatTensor(data)
            
        return tensor.to(self.device)
            
    def _from_torch(self, tensor):
        """
        Convert PyTorch tensor to NumPy array.
        
        Args:
            tensor: torch.Tensor
            
        Returns:
            numpy.ndarray
        """
        return tensor.cpu().numpy()
        
    def prepare_content_features(self):
        """
        Prepare article features using TF-IDF with GPU/MPS acceleration.
        """
        # Calculate popularity scores
        popularity = self.transactions_df['article_id'].value_counts()
        self.articles_df['popularity_score'] = (
            self.articles_df['article_id']
            .map(popularity)
            .fillna(0)
        )
        self.articles_df['popularity_normalized'] = (
            self.articles_df['popularity_score'] / 
            self.articles_df['popularity_score'].max()
        )

        # Define and process text columns
        text_columns = [
            'product_type_name',
            'product_group_name',
            'garment_group_name',
            'colour_group_name',
            'section_name',
            'perceived_colour_value_name',
            'perceived_colour_master_name',
            'detail_desc'
        ]

        for col in text_columns:
            if col in self.articles_df.columns:
                if pd.api.types.is_categorical_dtype(self.articles_df[col]):
                    self.articles_df[col] = self.articles_df[col].astype(str)
                self.articles_df[col] = self.articles_df[col].fillna('')
        
        # Combine text features
        df_text = self.articles_df[text_columns]
        self.articles_df['combined_features'] = (
            df_text.apply(' '.join, axis=1)
            .str.lower()
        )

        # Create TF-IDF matrix and convert to PyTorch
        self.feature_matrix = self.tfidf.fit_transform(
            self.articles_df['combined_features']
        )
        if self.device.type != 'cpu':
            self.feature_matrix_torch = self._to_torch(self.feature_matrix)

    def _torch_cosine_similarity(self, X, Y=None):
        """
        Calculate cosine similarity using PyTorch.
        Optimized for both CUDA and MPS backends.
        
        Args:
            X: Query matrix
            Y: Reference matrix (optional)
            
        Returns:
            Similarity matrix
        """
        if Y is None:
            Y = X
            
        # Convert to torch tensors if needed
        if not torch.is_tensor(X):
            X = self._to_torch(X)
        if not torch.is_tensor(Y):
            Y = self._to_torch(Y)
            
        # Normalize the matrices
        X_normalized = torch.nn.functional.normalize(X, p=2, dim=1)
        Y_normalized = torch.nn.functional.normalize(Y, p=2, dim=1)
        
        # Compute similarity with memory optimization for large matrices
        batch_size = 1024  # Adjust based on available memory
        if X.shape[0] * Y.shape[0] > batch_size * batch_size:
            similarities = torch.zeros(X.shape[0], Y.shape[0], device=self.device)
            for i in range(0, X.shape[0], batch_size):
                end_idx = min(i + batch_size, X.shape[0])
                batch_X = X_normalized[i:end_idx]
                batch_sim = torch.mm(batch_X, Y_normalized.t())
                similarities[i:end_idx] = batch_sim
            return similarities
        else:
            return torch.mm(X_normalized, Y_normalized.t())
        
    def recommend_items(
            self,
            customer_id,
            n_items=10,
            similarity_weight=0.3,
            customer_preference_weight=0.4,
            popularity_weight=0.3,
            filter_already_purchased=True  # New parameter
        ):
            """
            Generate personalized recommendations using available acceleration.
            All computations use float32 for MPS compatibility.
            
            Args:
                customer_id: ID of the customer
                n_items: Number of items to recommend
                similarity_weight: Weight for similarity scores
                customer_preference_weight: Weight for customer category preferences
                popularity_weight: Weight for item popularity
                filter_already_purchased: Whether to exclude items the customer has already purchased
            """
            # Get customer purchase history
            customer_history = self.transactions_df[
                self.transactions_df['customer_id'] == customer_id
            ]
            
            # Handle cold start case
            if customer_history.empty:
                return self._get_cold_start_recommendations(n_items)

            # Get customer preferences and recent items
            category_preferences = self._get_customer_preferences(customer_history)
            recent_items = (
                customer_history
                .sort_values('t_dat', ascending=False)['article_id']
                .tolist()
            )

            # Calculate similarity scores using acceleration
            feature_matrix = (self.feature_matrix_torch 
                            if self.device.type != 'cpu' 
                            else self._to_torch(self.feature_matrix))
            similarity_scores = torch.zeros(len(self.articles_df), 
                                        device=self.device, 
                                        dtype=torch.float32)
            
            for article_id in recent_items:
                article_idx = self.articles_df[
                    self.articles_df['article_id'] == article_id
                ].index[0]
                
                article_features = feature_matrix[article_idx:article_idx + 1]
                similarities = self._torch_cosine_similarity(article_features, feature_matrix)
                similarity_scores += similarities.squeeze()
                
            similarity_scores = similarity_scores / max(1, len(recent_items))

            # Calculate preference and popularity scores with explicit float32 dtype
            preference_scores = torch.tensor(
                self.articles_df['product_group_name']
                .map(category_preferences)
                .fillna(0)
                .values,
                device=self.device,
                dtype=torch.float32
            )
            
            popularity_scores = torch.tensor(
                self.articles_df['popularity_normalized'].values,
                device=self.device,
                dtype=torch.float32
            )

            # Combine scores with weights
            final_scores = (
                similarity_weight * similarity_scores +
                customer_preference_weight * preference_scores +
                popularity_weight * popularity_scores
            )

            # Exclude previously purchased items if requested
            if filter_already_purchased:
                purchased_indices = self.articles_df[
                    self.articles_df['article_id'].isin(customer_history['article_id'])
                ].index
                final_scores = self._from_torch(final_scores)
                final_scores[purchased_indices] = -1
            else:
                final_scores = self._from_torch(final_scores)

            # Get top recommendations
            top_indices = final_scores.argsort()[::-1][:n_items]
            top_articles = self.articles_df.iloc[top_indices]['article_id'].values
            top_scores = final_scores[top_indices]

            return list(zip(top_articles, top_scores))
            
        
    def _get_customer_preferences(self, customer_history):
        """Calculate customer category preferences based on purchase history."""
        purchases_with_info = customer_history.merge(
            self.articles_df[['article_id', 'product_group_name']],
            on='article_id'
        )
        category_counts = purchases_with_info['product_group_name'].value_counts()
        return (category_counts / len(purchases_with_info)).to_dict()

    def _get_cold_start_recommendations(self, n_recommendations):
        """Generate recommendations for new customers without purchase history."""
        popular_items = self.articles_df.sort_values(
            'popularity_normalized',
            ascending=False
        )
        diverse_recommendations = []
        seen_groups = set()

        for _, item in popular_items.iterrows():
            if item['product_group_name'] not in seen_groups:
                diverse_recommendations.append(
                    (item['article_id'], item['popularity_normalized'])
                )
                seen_groups.add(item['product_group_name'])
            if len(diverse_recommendations) >= n_recommendations:
                break

        return diverse_recommendations
    
