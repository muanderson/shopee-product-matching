import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from model import Shopeetransformer
from data_loader import ShopeeDataset, get_transforms

def get_embeddings(model, data_loader, device):
    """Generates embeddings for all items in the dataloader."""
    model.eval()
    all_embeddings = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Generating Embeddings"):
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            embeddings = model(images, input_ids, attention_mask)
            all_embeddings.append(embeddings.cpu().numpy())
            
    return np.concatenate(all_embeddings, axis=0)

def find_matches(df, embeddings, k_neighbors=50, similarity_threshold=0.85):
    """Finds matching products based on embedding similarity."""
    
    # --- ADDED ---
    # Dynamically adjust k to be no more than the number of samples
    num_samples = len(embeddings)
    if k_neighbors > num_samples:
        print(f"Warning: k_neighbors ({k_neighbors}) is greater than number of samples ({num_samples}). Adjusting to {num_samples}.")
        k_neighbors = num_samples
    # -------------

    # Using sklearn's NearestNeighbors for cosine distance
    nn_model = NearestNeighbors(n_neighbors=k_neighbors, metric='cosine', n_jobs=-1)
    nn_model.fit(embeddings)
    
    distances, indices = nn_model.kneighbors(embeddings)
    
    predictions = []
    for i in tqdm(range(len(embeddings)), desc="Finding Matches"):
        # Get the original posting_ids for the neighbors
        neighbor_indices = indices[i]
        neighbor_distances = distances[i]
        
        # Filter neighbors based on the similarity threshold
        # Similarity = 1 - Cosine Distance
        confident_neighbors = neighbor_indices[neighbor_distances < (1 - similarity_threshold)]
        
        # Get the posting_ids for the confident matches
        match_ids = df['posting_id'].iloc[confident_neighbors].tolist()
        
        predictions.append(" ".join(match_ids))
        
    df['matches'] = predictions
    return df

def main():
    config = {
        'model_path': r'C:\Users\Matthew\Documents\PhD\shopee-product-matching\models\best_model_fold_4.pt',
        'image_dir': r'C:\Users\Matthew\Documents\PhD\shopee-product-matching\data\test_images',
        'tabular_data_path': r'C:\Users\Matthew\Documents\PhD\shopee-product-matching\data\test.csv',
        'output_file': r'C:\Users\Matthew\Documents\PhD\shopee-product-matching\data\submission.csv',

        'image_size': 224,
        'batch_size': 128, 
        'embedding_size': 512,
        'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        
        # Post-processing parameters
        'k_neighbors': 50, # Find up to 50 nearest neighbors
        'similarity_threshold': 0.85 # The confidence threshold for a match
    }

    # --- 1. Load Model ---
    print("Loading model...")
    model = Shopeetransformer(embed_dim=config['embedding_size'])
    model.load_state_dict(torch.load(config['model_path'], map_location=config['device']))
    model.to(config['device'])
    print("Model loaded successfully.")

    # --- 2. Prepare Data ---
    print("Preparing test data...")
    df_test = pd.read_csv(config['tabular_data_path'])
    # Add image_path column
    df_test['image_path'] = df_test['image'].apply(
        lambda x: os.path.join(config['image_dir'], x)
    )
    
    # The test dataset from ShopeeDataset doesn't need labels
    test_dataset = ShopeeDataset(
        df_test, 
        transform=get_transforms(config['image_size'], is_training=False)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    print("Data prepared.")

    # --- 3. Generate Embeddings ---
    test_embeddings = get_embeddings(model, test_loader, config['device'])

    # --- 4. Find Matches using Post-Processing ---
    submission_df = find_matches(
        df_test, 
        test_embeddings,
        k_neighbors=config['k_neighbors'],
        similarity_threshold=config['similarity_threshold']
    )

    # --- 5. Create Submission File ---
    submission_df[['posting_id', 'matches']].to_csv(config['output_file'], index=False)
    print(f"Submission file created at: {config['output_file']}")

if __name__ == "__main__":
    main()