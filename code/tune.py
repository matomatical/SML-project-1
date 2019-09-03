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
bar_grid = tqdm(GRID, desc="Parameter combinations", position=2, dynamic_ncols=True)
for params in bar_grid:
    tqdm.write(f"Parameter combination: {params}")
    
    # begin experiment for this parameter combination:
    accuracies = []
    bar_folds = tqdm(FOLDS, desc="Cross-validation", position=1, leave=False, dynamic_ncols=True)
    for i, (train_ids, test_ids) in enumerate(bar_folds, start=1):
        # train and evalute the model on this data split:
        bar_train = tqdm(DATA[train_ids], desc="Training...", position=0, leave=False, dynamic_ncols=True)
        model = MODEL(bar_train, **params)
        bar_eval = tqdm(DATA[test_ids], desc="Evaluating...", position=0, leave=False, dynamic_ncols=True)
        accuracy, *_ = eval.evaluate(model, bar_eval)
        accuracies.append(accuracy)
        tqdm.write(f"Test {i} of {params} complete. Fold accuracy: {accuracy:.2%}")
        # and don't forget to record the results to the log file!
        with open(LOGFILENAME, 'a') as logfile:
            results = {'params': params, 'fold': i, 'accuracy': accuracy, 'time': time.time()}
            print(json.dumps(results), file=logfile, flush=True)

    # experiment done! compute the average result
    avg_accuracy = sum(accuracies)/len(accuracies)
    tqdm.write(f"Finished testing {params}. Mean accuracy from {N_SPLITS} folds: {avg_accuracy:.2%}")
