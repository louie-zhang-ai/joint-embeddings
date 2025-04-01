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


def bpr_loss(pos_scores, neg_scores):
    return -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()

# Example usage:
model = TwoTowerModel(num_users=10000, num_items=5000, embedding_dim=64)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Dummy training batch
batch_user = torch.randint(0, 10000, (32,))
batch_pos_item = torch.randint(0, 5000, (32,))
batch_neg_item = torch.randint(0, 5000, (32,))

model.train()
user_vecs, pos_vecs = model(batch_user, batch_pos_item)
_, neg_vecs = model(batch_user, batch_neg_item)

pos_scores = (user_vecs * pos_vecs).sum(dim=1)
neg_scores = (user_vecs * neg_vecs).sum(dim=1)

loss = bpr_loss(pos_scores, neg_scores)

optimizer.zero_grad()
loss.backward()
optimizer.step()

