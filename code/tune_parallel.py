import os
import sys
import multiprocessing as mp

from tqdm import tqdm

import tune


def main():
    # handle command line arguments
    try:
        n_processes = int(sys.argv[1])
        first_fold_id, folds, module_name, model_class, grid = tune.parse_args(sys.argv[2:])
    except:
        print("usage:", sys.argv[0], "<n_processes> <folds: i-j> <model> <hyper-parameter grid spec>")
        print("e.g.: ", sys.argv[0], "4 1-1 models.baseline.simple_ngram L=500 n=4 norm=False")
        sys.exit(1)
    
    # Where to save the results?
    # (default: experiments/MODULE_NAME.jsonl)
    logfilename = os.path.join(tune.LOGFILEDIR, module_name + ".jsonl")
 
    # generate list of experiment inputs
    # jobs :: [(params, (i, fold), model_class, LOGFILENAME), ...]
    jobs = []
    for params in grid:
        for i, fold in enumerate(folds, start=first_fold_id):
            jobs.append((params, (i, fold), model_class, logfilename))

    # let the PARALLEL science begin! #uwu
    with mp.Pool(processes=n_processes) as pool:
        results_generator = pool.imap_unordered(experiment_wrapper, jobs, chunksize=1)
        results_bar = tqdm(results_generator, total=len(jobs), dynamic_ncols=True)
        results_bar.set_description_str(f"{n_processes} cores experimenting...")
        for result in results_bar:
            (params, (i, _), *_), accuracy = result
            tqdm.write(f"Test of {params} on fold {i} complete. Fold accuracy: {accuracy:.2%}")
        pool.close()
        pool.join()
    
    print("Experiments complete! Run `python rank.py` to update the charts!")

def experiment_wrapper(job):
    accuracy = tune.experiment(*job, show_progress_bars=False)
    return job, accuracy


if __name__ == '__main__':
    main()
