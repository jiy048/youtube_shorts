"""
Synthetic data generator for YouTube Shorts recommendation system
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_users(n_users=10000):
    """Generate synthetic user data"""
    ages = np.random.normal(25, 8, n_users).clip(13, 65).astype(int)
    genders = np.random.choice(['M', 'F', 'Other'], n_users, p=[0.45, 0.45, 0.1])
    categories = ['Gaming', 'Music', 'Comedy', 'Sports', 'Education', 'Travel', 'Food', 'Fashion']
    preferred_categories = np.random.choice(categories, n_users)
    
    users_df = pd.DataFrame({
        'user_id': [f'user_{i}' for i in range(n_users)],
        'age': ages,
        'gender': genders,
        'preferred_category': preferred_categories,
        'join_date': pd.date_range('2020-01-01', '2024-12-01', periods=n_users)
    })
    return users_df

def generate_videos(n_videos=10000):
    """Generate synthetic video data"""
    categories = ['Gaming', 'Music', 'Comedy', 'Sports', 'Education', 'Travel', 'Food', 'Fashion']
    durations = np.random.uniform(15, 160, n_videos)  # 15-160 seconds for Shorts
    
    videos_df = pd.DataFrame({
        'video_id': [f'video_{i}' for i in range(n_videos)],
        'creator_id': np.random.randint(0, 1000, n_videos),
        'category': np.random.choice(categories, n_videos),
        'duration_sec': durations,
        'upload_date': pd.date_range('2020-01-01', '2024-12-01', periods=n_videos),
        'view_count': np.random.lognormal(10, 2, n_videos).astype(int)
    })
    return videos_df

def generate_interactions(users_df, videos_df, n_interactions=100000):
    """Generate synthetic user-video interactions"""
    
    # Sample users and videos
    user_ids = np.random.choice(users_df['user_id'], n_interactions)
    video_ids = np.random.choice(videos_df['video_id'], n_interactions)
    
    # Generate timestamps
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 1)
    timestamps = [start_date + timedelta(seconds=random.randint(0, int((end_date - start_date).total_seconds()))) 
                  for _ in range(n_interactions)]
    
    # Generate watch times (influenced by video duration)
    watch_times = []
    liked = []
    shared = []
    
    for i in range(n_interactions):
        video_duration = videos_df[videos_df['video_id'] == video_ids[i]]['duration_sec'].iloc[0]
        
        # Watch time: higher probability to watch more if video matches user preference
        user_pref = users_df[users_df['user_id'] == user_ids[i]]['preferred_category'].iloc[0]
        video_cat = videos_df[videos_df['video_id'] == video_ids[i]]['category'].iloc[0]
        
        if user_pref == video_cat:
            # Higher engagement for matching preferences
            watch_ratio = np.random.beta(3, 2)  # Skewed towards higher values
            like_prob = 0.15
            share_prob = 0.05
        else:
            # Lower engagement for non-matching
            watch_ratio = np.random.beta(2, 3)  # Skewed towards lower values
            like_prob = 0.05
            share_prob = 0.01
        
        watch_time = min(watch_ratio * video_duration, video_duration)
        watch_times.append(watch_time)
        
        # Engagement more likely with higher watch time
        engagement_multiplier = watch_time / video_duration
        liked.append(1 if np.random.random() < like_prob * (1 + engagement_multiplier) else 0)
        shared.append(1 if np.random.random() < share_prob * (1 + engagement_multiplier) else 0)
    
    interactions_df = pd.DataFrame({
        'user_id': user_ids,
        'video_id': video_ids,
        'watch_time': watch_times,
        'liked': liked,
        'shared': shared,
        'timestamp': timestamps
    })
    
    # Add some noise and edge cases
    # Some very short watches
    noise_indices = np.random.choice(len(interactions_df), size=int(0.1 * len(interactions_df)), replace=False)
    interactions_df.loc[noise_indices, 'watch_time'] = np.random.uniform(0, 5, len(noise_indices))
    
    # Ensure minimum engagement rates
    if interactions_df['liked'].mean() < 0.05:
        additional_likes = np.random.choice(len(interactions_df), 
                                          size=int(0.05 * len(interactions_df) - interactions_df['liked'].sum()), 
                                          replace=False)
        interactions_df.loc[additional_likes, 'liked'] = 1
    
    if interactions_df['shared'].mean() < 0.02:
        additional_shares = np.random.choice(len(interactions_df), 
                                           size=int(0.02 * len(interactions_df) - interactions_df['shared'].sum()), 
                                           replace=False)
        interactions_df.loc[additional_shares, 'shared'] = 1
    
    return interactions_df

def main():
    """Generate all synthetic datasets"""
    print("Generating synthetic data for YouTube Shorts recommendation system...")
    
    # Create data directory
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    # Generate datasets
    print("Generating users...")
    users_df = generate_users(10000)
    users_df.to_csv(data_dir / 'users.csv', index=False)
    print(f"Generated {len(users_df)} users")
    
    print("Generating videos...")
    videos_df = generate_videos(10000)
    videos_df.to_csv(data_dir / 'videos.csv', index=False)
    print(f"Generated {len(videos_df)} videos")
    
    print("Generating interactions...")
    interactions_df = generate_interactions(users_df, videos_df, 100000)
    interactions_df.to_csv(data_dir / 'interactions_cleaned.csv', index=False)
    print(f"Generated {len(interactions_df)} interactions")
    
    # Print summary statistics
    print("\n=== Data Summary ===")
    print(f"Users: {len(users_df)}")
    print(f"Videos: {len(videos_df)}")
    print(f"Interactions: {len(interactions_df)}")
    print(f"Like rate: {interactions_df['liked'].mean():.3f}")
    print(f"Share rate: {interactions_df['shared'].mean():.3f}")
    print(f"Avg watch time: {interactions_df['watch_time'].mean():.1f}s")
    
    # Category alignment
    merged = interactions_df.merge(users_df[['user_id', 'preferred_category']], on='user_id')
    merged = merged.merge(videos_df[['video_id', 'category']], on='video_id')
    alignment = (merged['preferred_category'] == merged['category']).mean()
    print(f"Category alignment: {alignment:.3f}")
    
    print("\nData generation complete!")

if __name__ == "__main__":
    main()