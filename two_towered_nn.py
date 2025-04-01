import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoTowerModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super(TwoTowerModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # Optionally, you can add MLP towers here
        self.user_mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

        self.item_mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

    def forward(self, user_ids, item_ids):
        user_vecs = self.user_embedding(user_ids)
        item_vecs = self.item_embedding(item_ids)

        user_vecs = self.user_mlp(user_vecs)
        item_vecs = self.item_mlp(item_vecs)

        # Normalize for cosine similarity if desired
        user_vecs = F.normalize(user_vecs, dim=1)
        item_vecs = F.normalize(item_vecs, dim=1)

        return user_vecs, item_vecs

    def predict(self, user_ids, item_ids):
        user_vecs, item_vecs = self.forward(user_ids, item_ids)
        return (user_vecs * item_vecs).sum(dim=1)  # dot product

