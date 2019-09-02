import os
import time
import json
import importlib

from tqdm import tqdm

import sklearn.model_selection as model_selection
import numpy as np

import data; data.load_train()
import eval

# which model to experiment with?
MODULE = "models.baseline.simple_ngram"
MODEL  = importlib.import_module(MODULE).Model

# what values of hyperparameters define the grid?
# keys: hyper parameter names, values: list of values for hyper parameters
GRID = model_selection.ParameterGrid({
    "L": [200, 500, 1000],
    "n": [2, 3, 4, 5]
})

# generate and fold training data
DATA = np.array(data.TRAIN)
N_SPLITS = 8
RANDOM = 51
FOLDS = list(model_selection.KFold(n_splits=N_SPLITS, random_state=RANDOM).split(DATA))

# where do results go?
LOGFILEDIR  = "experiments"
os.makedirs(LOGFILEDIR, exist_ok=True)
LOGFILENAME = os.path.join(LOGFILEDIR, MODULE + ".jsonl") # log results in JSON-line format


# Let the science begin! #uwu
for params in tqdm(GRID, desc="Parameter combinations", position=2):
    tqdm.write(f"Parameter combination: {params}")
    
    # begin experiment for this parameter combination:
    accuracies = []
    for i, (train_ids, test_ids) in enumerate(tqdm(FOLDS, desc="Cross-validation", position=1, leave=False), start=1):
        # train and evalute the model on this data split:
        model = MODEL(tqdm(DATA[train_ids], desc="training", position=0, leave=False), **params)
        accuracy, *_ = eval.evaluate(model, tqdm(DATA[test_ids], desc="evaluating", position=0, leave=False))
        accuracies.append(accuracy)
        tqdm.write(f"Test {i} of {params} complete. Fold accuracy: {accuracy:.2%}")
    avg_accuracy = sum(accuracies)/len(accuracies)
    tqdm.write(f"Finished testing {params}. Mean accuracy from {N_SPLITS} folds: {avg_accuracy:.2%}")
    # and don't forget to record the results to the log file!
    with open(LOGFILENAME, 'a') as logfile:
        results = {'params': params, 'accuracy': accuracy, 'accuracies': accuracies, 'time': time.time()}
        logfile.write(json.dumps(results)+"\n")
        logfile.flush()
