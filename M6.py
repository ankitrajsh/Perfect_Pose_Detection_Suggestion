import numpy as np
from scipy.sparse.linalg import svds

class PersonalizationRecommender:
    def __init__(self, num_users, num_items, k=20):
        self.num_users = num_users
        self.num_items = num_items
        self.k = k  # latent factors
        # Initialize user-item interaction matrix (e.g. ratings or implicit feedback)
        self.user_item_matrix = np.zeros((num_users, num_items))
        # After training, we get latent factors
        self.user_factors = None
        self.item_factors = None

    def update_interaction(self, user_id, item_id, rating=1):
        # Update feedback (e.g., user liked this outfit/pose)
        self.user_item_matrix[user_id, item_id] = rating

    def train(self):
        # Use SVD for matrix factorization (could replace with more complex DeepFM later)
        # Fill missing with zeros (implicit feedback)
        # Normalize matrix by subtracting mean user ratings (optional)
        # Here we do simple SVD decomposition
        u, s, vt = svds(self.user_item_matrix, k=self.k)
        self.user_factors = u
        self.item_factors = vt.T @ np.diag(s)

    def recommend(self, user_id, top_k=5):
        if self.user_factors is None or self.item_factors is None:
            raise ValueError("Model not trained. Call train() before recommending.")
        user_vector = self.user_factors[user_id, :]
        scores = self.item_factors @ user_vector
        # Rank items by descending score
        recommended_indices = np.argsort(scores)[::-1]
        return recommended_indices[:top_k], scores[recommended_indices[:top_k]]

if __name__ == "__main__":
    num_users = 10
    num_items = 50

    recommender = PersonalizationRecommender(num_users, num_items, k=10)

    # Simulate some user interactions
    recommender.update_interaction(user_id=0, item_id=2, rating=1)
    recommender.update_interaction(user_id=0, item_id=5, rating=1)
    recommender.update_interaction(user_id=0, item_id=10, rating=1)
    recommender.update_interaction(user_id=1, item_id=3, rating=1)
    recommender.update_interaction(user_id=1, item_id=5, rating=1)

    # Train model
    recommender.train()

    # Get top 5 recommendations for user 0
    items, scores = recommender.recommend(user_id=0, top_k=5)
    print("Top recommendations for user 0:")
    for i, score in zip(items, scores):
        print(f"Item {i} with score {score:.4f}")
