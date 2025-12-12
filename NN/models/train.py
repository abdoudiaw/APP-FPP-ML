import torch
from nn_learner import retrain

# Load in dataset
dataset='dataset_1d/solps_train.db'
# Train model
model =retrain(db_path=dataset)
# Save model
torch.save(model,f"diiid.pt")


