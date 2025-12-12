import torch
import sys
sys.path.append("../NN")
from nn_learner import LearnerModel, getAllGNDData, SOLVER_INDEXES
import plotter
import collections
import numpy as np
import matplotlib.pyplot as plt


# Load model
print("PYTORCH VERSION", torch.__version__)
with torch.serialization.safe_globals([LearnerModel]):
    model = torch.load("diiid.pt", weights_only=False)

# Load test dataset
DB_PATH='data/solps_test.db'
raw_dataset = getAllGNDData(DB_PATH)
OUTPUT_SLICE = SOLVER_INDEXES["output_slice"]

# Define inputs
Inputs = collections.namedtuple('Inputs', 'gas_puff p_tot core_flux dna hci r region')
Outputs = collections.namedtuple('Outputs', 'ne te ti po')
#Labels = ["$n_e$", "$T_e$", "$T_i$, "Potential"]
Labels = ["$n_e (m^{-3})$", "$T_e (eV)$", "$T_i (eV)$", "Potential (V)"]

# Define Inputs namedtuple
Inputs = collections.namedtuple('Inputs', 'gas_puff p_tot core_flux dna hci r region')

# Prepare containers
predictions = []
errbars = []

# Loop over dataset for individual predictions
for i, row in enumerate(raw_dataset):
    input_sample = Inputs(*row[:7])  # unpack inputs
    pred, err = model(input_sample)  # model returns unpacked Outputs
    predictions.append([pred.ne, pred.te, pred.ti, pred.po])
    errbars.append([err.ne, err.te, err.ti, err.po])


predictions = np.array(predictions)
errbars = np.array(errbars)

okay_prediction = model.iserrok(errbars)
point_badness = model.iserrok_fuzzy(errbars)
# Compute fuzzy error point-wise
point_badness = np.array([
    list(model.iserrok_fuzzy(Outputs(*err))) for err in errbars
])

n_targets = predictions.shape[1]
Labels = ["$n_e (m^{-3})$", "$T_e (eV)$", "$T_i (eV)$", "Potential (V)"]

print("Bad points:", (point_badness >= 1).any(axis=1).sum(axis=0))
true = raw_dataset[:, OUTPUT_SLICE]

plotter.plot_true_vs_pred_with_uncertainty(predictions, true, point_badness, model.err_info,  ["$n_e (m^{-3})$", "$T_e (eV)$", "$T_i (eV)$", "Potential (V)"], scale=4)
