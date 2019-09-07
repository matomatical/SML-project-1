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
    first_fold_id, folds, module_name, model_class, grid = parse_args(sys.argv[1:])

    # Where to save the results?
    # (default: experiments/MODULE_NAME.jsonl)
    logfilename = os.path.join(LOGFILEDIR, module_name + ".jsonl")

    # Let the science begin! #uwu
    bar_grid = tqdm(grid, desc="Parameter combinations", position=2, dynamic_ncols=True)
    for params in bar_grid:
        tqdm.write(f"Parameter combination: {params}")
        
        # begin experiments for this parameter combination:
        accuracies = []
        bar_folds = tqdm(folds, desc="Cross-validation", position=1, leave=False, dynamic_ncols=True)
        for i, fold in enumerate(bar_folds, start=first_fold_id):
            accuracy = experiment(params, (i, fold), model_class, logfilename)
            tqdm.write(f"Test {i} of {params} complete. Fold accuracy: {accuracy:.2%}")
            accuracies.append(accuracy)
        # experiments done! compute the average result over folds
        avg_accuracy = sum(accuracies)/len(accuracies)
        tqdm.write(f"Finished testing {params}. Mean accuracy from {len(folds)} folds: {avg_accuracy:.2%}")

def parse_args(args):
    # which folds should we test? shallow or deep search?
    first_fold_id, last_fold_id = map(int, args[0].split('-'))
    print("Experimenting with folds", first_fold_id, "to", last_fold_id, "inclusive")
    folds = FOLDS[first_fold_id-1:last_fold_id] # from 1-based to 0-based
    
    # which model to experiment with?
    module_name = args[1]
    print("Model:", module_name)
    model_class = importlib.import_module(module_name).Model

    # what values of hyperparameters define the grid?
    # keys: hyper parameter names, values: list of values for hyper parameters
    # turns ["L=300,400,500", "n=3,4,5"] into a {"L": ["300", ...], "n": ["3", ...]}
    grid_spec = {arg.split("=")[0]: arg.split("=")[1].split(",") for arg in args[2:]}
    print("Grid spec", grid_spec)
    grid = model_selection.ParameterGrid(grid_spec)

    return first_fold_id, folds, module_name, model_class, grid

def experiment(params, fold, model_class, logfilename, show_progress_bars=True):
    # unpack the data split for this experiment
    fold_id, (train_ids, test_ids) = fold

    train_data, test_data = DATA[train_ids], DATA[test_ids]

    # train and evalute the model on this data split:
    if show_progress_bars:
        train_data = tqdm(train_data, desc="Training...", position=0, leave=False, dynamic_ncols=True)
    model = model_class(train_data, **params)
    if show_progress_bars:
        test_data = tqdm(test_data, desc="Evaluating...", position=0, leave=False, dynamic_ncols=True)
    accuracy, *_ = eval.evaluate(model, test_data)

    # don't forget to record the results to the log file!
    with open(logfilename, 'a') as logfile:
        results = {'params': params, 'fold': fold_id, 'accuracy': accuracy, 'time': time.time()}
        print(json.dumps(results), file=logfile, flush=True)

    return accuracy

if __name__ == '__main__':
    main()
