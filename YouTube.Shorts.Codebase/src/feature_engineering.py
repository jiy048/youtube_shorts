"""
Feature engineering pipeline for YouTube Shorts recommendation system
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.decomposition import TruncatedSVD
import pickle
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.embeddings = {}
        
    def create_temporal_features(self, df):
        """Create time-based features"""
        logger.info("Creating temporal features...")
        df = df.copy()
        
        # Ensure timestamp is datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Extract time features
        df['hour_of_day'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['month'] = df['timestamp'].dt.month
        
        # Create day period categories
        df['day_period'] = pd.cut(df['hour_of_day'], 
                                 bins=[0, 6, 12, 18, 24], 
                                 labels=['night', 'morning', 'afternoon', 'evening'],
                                 include_lowest=True)
        
        # Calculate recency (days since interaction)
        max_date = df['timestamp'].max()
        df['recency_days'] = (max_date - df['timestamp']).dt.days
        
        return df
    
    def create_user_features(self, interactions_df, users_df):
        """Create user-level aggregated features"""
        logger.info("Creating user features...")
        
        # User interaction statistics
        user_stats = interactions_df.groupby('user_id').agg({
            'watch_time': ['mean', 'std', 'count'],
            'liked': ['mean', 'sum'],
            'shared': ['mean', 'sum'],
            'timestamp': ['min', 'max']
        }).round(4)
        
        # Flatten column names
        user_stats.columns = ['_'.join(col) for col in user_stats.columns]
        user_stats = user_stats.reset_index()
        
        # Calculate user tenure
        user_stats['user_tenure_days'] = (
            user_stats['timestamp_max'] - user_stats['timestamp_min']
        ).dt.days
        
        # User activity level
        user_stats['interactions_per_day'] = (
            user_stats['watch_time_count'] / (user_stats['user_tenure_days'] + 1)
        )
        
        # Merge with user demographics
        user_features = users_df.merge(user_stats, on='user_id', how='left')
        
        # Fill NaN values for users with no interactions
        numeric_cols = user_features.select_dtypes(include=[np.number]).columns
        user_features[numeric_cols] = user_features[numeric_cols].fillna(0)
        
        return user_features
    
    def create_video_features(self, interactions_df, videos_df):
        """Create video-level aggregated features"""
        logger.info("Creating video features...")
        
        # Video interaction statistics
        video_stats = interactions_df.groupby('video_id').agg({
            'watch_time': ['mean', 'std', 'count'],
            'liked': ['mean', 'sum'],
            'shared': ['mean', 'sum'],
            'timestamp': ['min', 'max']
        }).round(4)
        
        # Flatten column names
        video_stats.columns = ['_'.join(col) for col in video_stats.columns]
        video_stats = video_stats.reset_index()
        
        # Video popularity metrics
        video_stats['video_popularity'] = video_stats['watch_time_count']
        video_stats['engagement_rate'] = (
            video_stats['liked_sum'] + video_stats['shared_sum']
        ) / video_stats['watch_time_count']
        
        # Video age in days
        max_date = interactions_df['timestamp'].max()
        video_stats['video_age_days'] = (
            max_date - video_stats['timestamp_min']
        ).dt.days
        
        # Merge with video metadata
        video_features = videos_df.merge(video_stats, on='video_id', how='left')
        
        # Fill NaN values for videos with no interactions
        numeric_cols = video_features.select_dtypes(include=[np.number]).columns
        video_features[numeric_cols] = video_features[numeric_cols].fillna(0)
        
        return video_features
    
    def create_interaction_features(self, df, videos_df):
        """Create features from interaction data"""
        logger.info("Creating interaction features...")
        df = df.copy()
        
        # Merge with video duration
        df = df.merge(videos_df[['video_id', 'duration_sec']], on='video_id')
        
        # Watch completion rate
        df['completion_rate'] = np.clip(df['watch_time'] / df['duration_sec'], 0, 1)
        
        # Normalize watch time
        if 'watch_time_normalized' not in df.columns:
            scaler = MinMaxScaler()
            df['watch_time_normalized'] = scaler.fit_transform(df[['watch_time']])
            self.scalers['watch_time'] = scaler
        
        # Normalize recency
        if 'recency_days' in df.columns:
            if 'recency_normalized' not in df.columns:
                scaler = MinMaxScaler()
                df['recency_normalized'] = scaler.fit_transform(df[['recency_days']])
                self.scalers['recency'] = scaler
        
        return df
    
    def create_embeddings(self, interactions_df, n_components=50):
        """Create user and video embeddings using SVD"""
        logger.info(f"Creating embeddings with {n_components} components...")
        
        # Create user and video encoders
        user_encoder = LabelEncoder()
        video_encoder = LabelEncoder()
        
        # Encode user and video IDs
        interactions_df = interactions_df.copy()
        interactions_df['user_idx'] = user_encoder.fit_transform(interactions_df['user_id'])
        interactions_df['video_idx'] = video_encoder.fit_transform(interactions_df['video_id'])
        
        # Create interaction matrix
        interaction_matrix = pd.pivot_table(
            interactions_df, 
            values='liked', 
            index='user_idx', 
            columns='video_idx',
            fill_value=0
        )
        
        # Apply SVD
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        user_embeddings = svd.fit_transform(interaction_matrix)
        video_embeddings = svd.components_.T
        
        # Store encoders and embeddings
        self.encoders['user_encoder'] = user_encoder
        self.encoders['video_encoder'] = video_encoder
        self.embeddings['user_embeddings'] = user_embeddings
        self.embeddings['video_embeddings'] = video_embeddings
        self.embeddings['svd_model'] = svd
        
        logger.info(f"Created embeddings for {len(user_encoder.classes_)} users and {len(video_encoder.classes_)} videos")
        
        return user_embeddings, video_embeddings
    
    def add_embedding_features(self, df):
        """Add embedding features to dataframe"""
        if 'user_embeddings' not in self.embeddings:
            logger.warning("Embeddings not created yet. Call create_embeddings first.")
            return df
        
        logger.info("Adding embedding features...")
        df = df.copy()
        
        user_encoder = self.encoders['user_encoder']
        video_encoder = self.encoders['video_encoder']
        user_embeddings = self.embeddings['user_embeddings']
        video_embeddings = self.embeddings['video_embeddings']
        
        # Map user and video IDs to embeddings
        user_embedding_map = dict(zip(user_encoder.classes_, user_embeddings))
        video_embedding_map = dict(zip(video_encoder.classes_, video_embeddings))
        
        # Add embeddings (with fallback for cold start)
        n_components = user_embeddings.shape[1]
        
        def get_user_embedding(user_id):
            return user_embedding_map.get(user_id, np.zeros(n_components))
        
        def get_video_embedding(video_id):
            return video_embedding_map.get(video_id, np.zeros(n_components))
        
        # Add embedding columns
        user_emb_cols = [f'user_emb_{i}' for i in range(n_components)]
        video_emb_cols = [f'video_emb_{i}' for i in range(n_components)]
        
        user_embeddings_df = pd.DataFrame(
            [get_user_embedding(uid) for uid in df['user_id']],
            columns=user_emb_cols,
            index=df.index
        )
        
        video_embeddings_df = pd.DataFrame(
            [get_video_embedding(vid) for vid in df['video_id']],
            columns=video_emb_cols,
            index=df.index
        )
        
        df = pd.concat([df, user_embeddings_df, video_embeddings_df], axis=1)
        
        return df
    
    def create_sessionization_features(self, df, session_gap_minutes=30):
        """Create session-based features"""
        logger.info("Creating sessionization features...")
        df = df.copy()
        
        # Sort by user and timestamp
        df = df.sort_values(['user_id', 'timestamp'])
        
        # Calculate time gaps between interactions
        df['time_gap'] = df.groupby('user_id')['timestamp'].diff()
        
        # Create session boundaries (gap > session_gap_minutes)
        session_boundary = df['time_gap'] > pd.Timedelta(minutes=session_gap_minutes)
        df['session_id'] = session_boundary.groupby(df['user_id']).cumsum()
        
        # Create session-level features
        session_stats = df.groupby(['user_id', 'session_id']).agg({
            'watch_time': ['sum', 'count', 'mean'],
            'liked': 'sum',
            'shared': 'sum',
            'timestamp': ['min', 'max']
        })
        
        # Flatten column names
        session_stats.columns = ['_'.join(col) for col in session_stats.columns]
        session_stats = session_stats.reset_index()
        
        # Calculate session duration
        session_stats['session_duration_minutes'] = (
            session_stats['timestamp_max'] - session_stats['timestamp_min']
        ).dt.total_seconds() / 60
        
        # Map back to original dataframe
        df = df.merge(session_stats[['user_id', 'session_id', 'watch_time_count', 'session_duration_minutes']], 
                     on=['user_id', 'session_id'], suffixes=('', '_session'))
        
        # Rename for clarity
        df['session_length'] = df['watch_time_count_session']
        df = df.drop('watch_time_count_session', axis=1)
        
        return df
    
    def encode_categorical_features(self, df, categorical_cols=None):
        """Encode categorical features"""
        logger.info("Encoding categorical features...")
        df = df.copy()
        
        if categorical_cols is None:
            categorical_cols = ['gender', 'category', 'preferred_category', 'day_period']
        
        # Filter columns that exist in the dataframe
        existing_categorical_cols = [col for col in categorical_cols if col in df.columns]
        
        if existing_categorical_cols:
            # One-hot encode categorical variables
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoded_features = encoder.fit_transform(df[existing_categorical_cols])
            
            # Create feature names
            feature_names = encoder.get_feature_names_out(existing_categorical_cols)
            
            # Create dataframe with encoded features
            encoded_df = pd.DataFrame(encoded_features, columns=feature_names, index=df.index)
            
            # Concatenate with original dataframe
            df = pd.concat([df, encoded_df], axis=1)
            
            # Store encoder
            self.encoders['categorical_encoder'] = encoder
            
            logger.info(f"Encoded {len(existing_categorical_cols)} categorical features into {len(feature_names)} binary features")
        
        return df
    
    def create_advanced_features(self, df):
        """Create advanced engineered features"""
        logger.info("Creating advanced features...")
        df = df.copy()
        
        # Interaction features
        if 'hour_of_day' in df.columns and 'is_weekend' in df.columns:
            df['hour_weekend_interaction'] = df['hour_of_day'] * df['is_weekend']
        
        # Popularity-based features
        if 'video_popularity' in df.columns and 'watch_time' in df.columns:
            df['popularity_watch_ratio'] = df['video_popularity'] / (df['watch_time'] + 1e-6)
        
        # Engagement momentum features
        if 'liked' in df.columns and 'shared' in df.columns:
            df['total_engagement'] = df['liked'] + df['shared']
            
            # Rolling engagement features (last 5 interactions per user)
            df['user_recent_engagement'] = (
                df.groupby('user_id')['total_engagement']
                .rolling(window=5, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
        
        # Diversity features (if category information is available)
        if 'category' in df.columns:
            # User's category diversity
            user_category_counts = df.groupby('user_id')['category'].nunique()
            df['user_category_diversity'] = df['user_id'].map(user_category_counts)
        
        return df
    
    def process_features(self, interactions_df, users_df, videos_df):
        """Main feature processing pipeline"""
        logger.info("Starting feature engineering pipeline...")
        
        # Start with interactions
        df = interactions_df.copy()
        
        # 1. Create temporal features
        df = self.create_temporal_features(df)
        
        # 2. Create user and video features
        user_features = self.create_user_features(interactions_df, users_df)
        video_features = self.create_video_features(interactions_df, videos_df)
        
        # 3. Merge with user and video features
        df = df.merge(users_df[['user_id', 'age', 'gender', 'preferred_category']], on='user_id')
        df = df.merge(videos_df[['video_id', 'category', 'duration_sec']], on='video_id')
        
        # 4. Create interaction features
        df = self.create_interaction_features(df, videos_df)
        
        # 5. Add video popularity
        video_popularity = interactions_df['video_id'].value_counts().to_dict()
        df['video_popularity'] = df['video_id'].map(video_popularity).fillna(0)
        
        # 6. Create sessionization features
        df = self.create_sessionization_features(df)
        
        # 7. Create embeddings
        self.create_embeddings(interactions_df)
        
        # 8. Add embedding features
        df = self.add_embedding_features(df)
        
        # 9. Encode categorical features
        df = self.encode_categorical_features(df)
        
        # 10. Create advanced features
        df = self.create_advanced_features(df)
        
        # 11. Create target variable
        df['label'] = df['liked']  # Primary target
        df['label_engagement'] = ((df['liked'] == 1) | (df['shared'] == 1)).astype(int)  # Alternative target
        df['label_completion'] = (df['completion_rate'] > 0.5).astype(int)  # Completion-based target
        
        logger.info(f"Feature engineering complete. Final dataset shape: {df.shape}")
        
        return df
    
    def save_artifacts(self, save_dir='models'):
        """Save feature engineering artifacts"""
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        # Save scalers
        with open(save_path / 'scalers.pkl', 'wb') as f:
            pickle.dump(self.scalers, f)
        
        # Save encoders
        with open(save_path / 'encoders.pkl', 'wb') as f:
            pickle.dump(self.encoders, f)
        
        # Save embeddings
        with open(save_path / 'embeddings.pkl', 'wb') as f:
            pickle.dump(self.embeddings, f)
        
        logger.info(f"Feature engineering artifacts saved to {save_path}")
    
    def load_artifacts(self, save_dir='models'):
        """Load feature engineering artifacts"""
        save_path = Path(save_dir)
        
        # Load scalers
        try:
            with open(save_path / 'scalers.pkl', 'rb') as f:
                self.scalers = pickle.load(f)
        except FileNotFoundError:
            logger.warning("Scalers file not found")
        
        # Load encoders
        try:
            with open(save_path / 'encoders.pkl', 'rb') as f:
                self.encoders = pickle.load(f)
        except FileNotFoundError:
            logger.warning("Encoders file not found")
        
        # Load embeddings
        try:
            with open(save_path / 'embeddings.pkl', 'rb') as f:
                self.embeddings = pickle.load(f)
        except FileNotFoundError:
            logger.warning("Embeddings file not found")
        
        logger.info(f"Feature engineering artifacts loaded from {save_path}")

def main():
    """Main execution function"""
    # Load data
    users_df = pd.read_csv('data/users.csv')
    videos_df = pd.read_csv('data/videos.csv')
    interactions_df = pd.read_csv('data/interactions_cleaned.csv')
    
    # Initialize feature engineer
    fe = FeatureEngineer()
    
    # Process features
    processed_df = fe.process_features(interactions_df, users_df, videos_df)
    
    # Save processed data
    processed_df.to_csv('data/processed_interactions.csv', index=False)
    
    # Save artifacts
    fe.save_artifacts()
    
    logger.info("Feature engineering complete!")

if __name__ == "__main__":
    main()