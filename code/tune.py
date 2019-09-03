import os
import sys
import time
import json
import importlib

from tqdm import tqdm

import sklearn.model_selection as model_selection
import numpy as np

import data
import eval


# import the specified model:

# generate and fold training data
data.load_train()
DATA = np.array(data.TRAIN)
N_SPLITS = 8
RANDOM = 51
FOLDS = list(model_selection.KFold(n_splits=N_SPLITS, random_state=RANDOM).split(DATA))

# where do results go?
LOGFILEDIR  = "experiments"
os.makedirs(LOGFILEDIR, exist_ok=True)


def main():    
    # which folds should we test? shallow or deep search?
    first_fold, last_fold = map(int, sys.argv[1].split('-'))
    print("Experimenting with folds", first_fold, "to", last_fold, "inclusive")
    folds = FOLDS[first_fold-1:last_fold]
    
    # which model to experiment with?
    module_name = sys.argv[2]
    print("Model:", module_name)
    model_class = importlib.import_module(module_name).Model

    # what values of hyperparameters define the grid?
    # keys: hyper parameter names, values: list of values for hyper parameters
    # turns ["L=300,400,500", "n=3,4,5"] into a {"L": ["300", ...], "n": ["3", ...]}
    grid_spec = {arg.split("=")[0]: arg.split("=")[1].split(",") for arg in sys.argv[3:]}
    print("Grid spec", grid_spec)
    grid = model_selection.ParameterGrid(grid_spec)
    

    # Where to save the results?
    LOGFILENAME = os.path.join(LOGFILEDIR, module_name + ".jsonl")
    # (default: experiments/MODULE_NAME.jsonl)
    # ----------------------------------------------------------------------- #

    # Let the science begin! #uwu
    bar_grid = tqdm(grid, desc="Parameter combinations", position=2, dynamic_ncols=True)
    for params in bar_grid:
        tqdm.write(f"Parameter combination: {params}")
        
        # begin experiments for this parameter combination:
        accuracies = []
        bar_folds = tqdm(folds, desc="Cross-validation", position=1, leave=False, dynamic_ncols=True)
        for i, fold in enumerate(bar_folds, start=first_fold):
            accuracy = experiment(params, (i, fold), model_class, LOGFILENAME)
            tqdm.write(f"Test {i} of {params} complete. Fold accuracy: {accuracy:.2%}")
            accuracies.append(accuracy)
        # experiments done! compute the average result over folds
        avg_accuracy = sum(accuracies)/len(accuracies)
        tqdm.write(f"Finished testing {params}. Mean accuracy from {N_SPLITS} folds: {avg_accuracy:.2%}")

def experiment(params, fold, model_class, logfilename):
    # unpack the data split for this experiment
    fold_id, (train_ids, test_ids) = fold

    # train and evalute the model on this data split:
    bar_train = tqdm(DATA[train_ids], desc="Training...", position=0, leave=False, dynamic_ncols=True)
    model = model_class(bar_train, **params)
    bar_eval = tqdm(DATA[test_ids], desc="Evaluating...", position=0, leave=False, dynamic_ncols=True)
    accuracy, *_ = eval.evaluate(model, bar_eval)

    # don't forget to record the results to the log file!
    with open(logfilename, 'a') as logfile:
        results = {'params': params, 'fold': fold_id, 'accuracy': accuracy, 'time': time.time()}
        print(json.dumps(results), file=logfile, flush=True)

    return accuracy

if __name__ == '__main__':
    main()
