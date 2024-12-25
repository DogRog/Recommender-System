import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class NumericalCBF:
    def __init__(self, device='mps', batch_size=256, output_dir='recommendations'):
        self.device = self._get_device(device)
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.scaler = StandardScaler()
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
    def _get_device(self, device):
        if device is None:
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                return torch.device('cpu')
        return torch.device(device)
        
    def _get_timestamp(self):
        """Get formatted timestamp for filenames"""
        return datetime.now().strftime('%Y%m%d_%H%M%S')
        
    def fit(self, transactions_df, customers_df, articles_df):
        """Optimized fit method with numpy arrays for fast lookup"""
        print("Fitting the model...")

        # Store article IDs as numpy array for fast lookup
        self.article_ids = articles_df['article_id'].values
        self.article_id_to_idx = {aid: idx for idx, aid in enumerate(self.article_ids)}

        # Pre-compute user purchase histories as sets
        print("Pre-computing user purchase histories...")
        self.user_histories = {}
        for user_id, group in tqdm(transactions_df.groupby('customer_id'), desc="Building history lookup"):
            article_indices = [self.article_id_to_idx[aid] for aid in group['article_id']]
            self.user_histories[user_id] = set(article_indices)

        # Calculate popularity scores once and store as class attribute
        popularity = transactions_df['article_id'].value_counts()
        self.popularity_scores = np.zeros(len(self.article_ids))
        for article_id, score in popularity.items():
            if article_id in self.article_id_to_idx:
                self.popularity_scores[self.article_id_to_idx[article_id]] = score

        price = transactions_df.groupby('article_id')['price'].mean()

        # Store popularity scores as numpy array
        articles_df['popularity_score'] = articles_df['article_id'].map(popularity)
        articles_df['price'] = articles_df['article_id'].map(price)

        print("Processing features...")
        numerical_columns = [
            'product_type_no',
            'garment_group_no',
            'colour_group_code',
            'section_no',
            'perceived_colour_value_id',
            'perceived_colour_master_id',
            'price',
            'popularity_score'
        ]

        # Create feature matrix
        feature_matrix = articles_df[numerical_columns].fillna(0).values

        # Scale features and convert to tensor
        scaled_features = self.scaler.fit_transform(feature_matrix)
        self.feature_tensor = torch.FloatTensor(scaled_features).to(self.device)

        print("Model fitted successfully!")
        return self
        
    def batch_recommend_items(self, user_ids, n_items=10, filter_already_purchased=True, chunk_size=1000):
        """Generate recommendations in batches and save to CSV"""
        print(f"\nGenerating recommendations for {len(user_ids)} users...")
        
        # Create timestamp for this batch of recommendations
        timestamp = self._get_timestamp()
        output_file = os.path.join(self.output_dir, f'recommendations_{timestamp}.csv')
        
        # Create CSV file with headers
        with open(output_file, 'w') as f:
            f.write('user_id,rank,article_id,score\n')
        
        # Process users in chunks to save memory
        for chunk_start in tqdm(range(0, len(user_ids), chunk_size), desc="Processing user chunks"):
            chunk_end = min(chunk_start + chunk_size, len(user_ids))
            chunk_users = user_ids[chunk_start:chunk_end]
            
            # Calculate user profiles for chunk
            batch_profiles = []
            cold_start_users = []
            
            for user_id in tqdm(chunk_users, desc="Building user profiles", leave=False):
                history = self.user_histories.get(user_id, set())
                if not history:
                    cold_start_users.append(user_id)
                    batch_profiles.append(torch.zeros(self.feature_tensor.shape[1], device=self.device))
                    continue
                    
                profile = self.feature_tensor[list(history)].mean(dim=0)
                batch_profiles.append(profile)
            
            # Convert profiles to tensor
            user_profiles = torch.stack(batch_profiles)
            
            # Calculate similarities in batches
            n_users = len(chunk_users)
            all_similarities = []
            
            for i in tqdm(range(0, n_users, self.batch_size), desc="Computing similarities", leave=False):
                end_idx = min(i + self.batch_size, n_users)
                batch_profiles = user_profiles[i:end_idx]
                
                similarities = torch.nn.functional.cosine_similarity(
                    batch_profiles.unsqueeze(1),
                    self.feature_tensor.unsqueeze(0),
                    dim=2
                )
                all_similarities.append(similarities)
            
            # Combine similarities
            all_similarities = torch.cat(all_similarities, dim=0).cpu().numpy()
            
            # Generate and save recommendations for chunk
            print(f"Saving recommendations for users {chunk_start} to {chunk_end}")
            
            with open(output_file, 'a') as f:
                for idx, user_id in enumerate(chunk_users):
                    if user_id in cold_start_users:
                        # Use pre-computed popularity scores for cold start
                        top_indices = np.argsort(self.popularity_scores)[::-1][:n_items]
                        scores = self.popularity_scores[top_indices]
                    else:
                        similarities = all_similarities[idx]
                        if filter_already_purchased:
                            similarities[list(self.user_histories[user_id])] = -1
                        
                        top_indices = np.argpartition(similarities, -n_items)[-n_items:]
                        top_indices = top_indices[np.argsort(similarities[top_indices])][::-1]
                        scores = similarities[top_indices]
                    
                    # Write recommendations to CSV
                    for rank, (article_idx, score) in enumerate(zip(top_indices, scores)):
                        f.write(f'{user_id},{rank+1},{self.article_ids[article_idx]},{score:.6f}\n')
        
        print(f"\nRecommendations saved to: {output_file}")
        return output_file
    
    def recommend_items(self, user_id, n_items=10, filter_already_purchased=True):
        """Single user recommendations"""
        # Get user profile
        history = self.user_histories.get(user_id, set())
        if not history:
            # Cold start case - return popular items
            top_indices = np.argsort(self.popularity_scores)[::-1][:n_items]
            scores = self.popularity_scores[top_indices]
            return [(int(self.article_ids[idx]), float(score)) 
                    for idx, score in zip(top_indices, scores)]
        
        # Compute user profile
        profile = self.feature_tensor[list(history)].mean(dim=0)
        
        # Calculate similarities
        similarities = torch.nn.functional.cosine_similarity(
            profile.unsqueeze(0),
            self.feature_tensor,
            dim=1
        ).cpu().numpy()
        
        # Filter purchased items if requested
        if filter_already_purchased:
            similarities[list(history)] = -1
        
        # Get top items
        top_indices = np.argpartition(similarities, -n_items)[-n_items:]
        top_indices = top_indices[np.argsort(similarities[top_indices])][::-1]
        scores = similarities[top_indices]
        
        # Return recommendations as list of tuples
        return [(self.article_ids[idx], float(score)) 
                for idx, score in zip(top_indices, scores)]