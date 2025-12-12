import torch
import sys
sys.path.append("../nn")
from nn_learner import retrain

# Load in dataset
dataset='data/solps_train.db'
# Train model
model =retrain(db_path=dataset)
# Save model
torch.save(model,f"diiid.pt")


