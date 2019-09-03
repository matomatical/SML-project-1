import os
import time
import json
import importlib
import multiprocessing as mp

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
    # Define the experiments here: ------------------------------------------ #
    # which model to experiment with?
    MODULE = "models.baseline.simple_ngram"

    # what values of hyperparameters define the grid?
    # keys: hyper parameter names, values: list of values for hyper parameters
    GRID = model_selection.ParameterGrid({
        "L": [300, 400, 500, 600, 700],
        "n": [6]
    })

    # Where to save the results?
    LOGFILENAME = os.path.join(LOGFILEDIR, MODULE + ".jsonl")
    # (default: experiments/MODULE_NAME.jsonl)
    # ----------------------------------------------------------------------- #

    # Let the science begin! #uwu
    model_class = importlib.import_module(MODULE).Model
    # bar_grid = tqdm(GRID, desc="Parameter combinations", position=2, dynamic_ncols=True)
    # generate input
    # (params, (i, fold), model_class, LOGFILENAME)
    jobs = []
    for params in GRID:
        # tqdm.write(f"Parameter combination: {params}")
        
        # begin experiments for this parameter combination:
        # accuracies = []
        # bar_folds = tqdm(FOLDS, desc="Cross-validation", position=1, leave=False, dynamic_ncols=True)
        for i, fold in enumerate(FOLDS, start=1):
            jobs.append((params, (i, fold), model_class, LOGFILENAME))
            # accuracy = experiment(params, (i, fold), model_class, LOGFILENAME)
            # tqdm.write(f"Test {i} of {params} complete. Fold accuracy: {accuracy:.2%}")
            # accuracies.append(accuracy)
        # experiments done! compute the average result over folds
        # avg_accuracy = sum(accuracies)/len(accuracies)
        # tqdm.write(f"Finished testing {params}. Mean accuracy from {N_SPLITS} folds: {avg_accuracy:.2%}")

    pool = mp.Pool(processes=4)

    results = list(pool.imap_unordered(experiment_wrapper, jobs, chunksize=1))
    pool.close()
    pool.join()
    
    print(results)

def experiment_wrapper(job):
    experiment(*job)

def experiment(params, fold, model_class, logfilename):
    # unpack the data split for this experiment
    fold_id, (train_ids, test_ids) = fold

    # train and evalute the model on this data split:
    # bar_train = tqdm(DATA[train_ids], desc="Training...", position=0, leave=False, dynamic_ncols=True)
    model = model_class(DATA[train_ids], **params)
    # bar_eval = tqdm(DATA[test_ids], desc="Evaluating...", position=0, leave=False, dynamic_ncols=True)
    accuracy, *_ = eval.evaluate(model, DATA[test_ids])

    # don't forget to record the results to the log file!
    with open(logfilename, 'a') as logfile:
        results = {'params': params, 'fold': fold_id, 'accuracy': accuracy, 'time': time.time()}
        print(json.dumps(results), file=logfile, flush=True)

    return accuracy

if __name__ == '__main__':
    main()
