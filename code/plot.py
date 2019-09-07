import sys
import json
from statistics import mean
from collections import defaultdict
import numpy as np
from cycler import cycler
import matplotlib.pyplot as plt

def main():
    filename = sys.argv[1]

    # load the data
    results = []
    with open(filename) as infile:
        results = [json.loads(line) for line in infile]

    # collate folds
    collated_results = defaultdict(list)
    for result in results:
        params = tuple(result['params'].items())
        collated_results[params].append((result['fold'], result['accuracy']))
    scored_results = defaultdict(dict)
    full_params = {}
    for params, fold_accs in collated_results.items():
        L = [int(val) for key, val in params if key == "L"][0]
        desc = ', '.join('='.join(map(str, p)) for p in params if p[0] != "L")
        folds, accs = zip(*fold_accs)
        scored_results[desc][L] = mean(accs)
        full_params[desc] = dict(params)

    proc_cycler = ( cycler(color=plt.rcParams['axes.prop_cycle'].by_key()['color'])
                  * cycler(linestyle=['-','--','-.',':']))
    plt.rc('axes', prop_cycle=proc_cycler)
    plt.gcf().set_size_inches(8, 8.5)
    for desc, data in sorted(scored_results.items()):
        params = full_params[desc]
        marker = '$' + params.get("level", "char")[0] + '$'
        plt.plot(*zip(*sorted(data.items())), marker=marker, label=desc)
    plt.gca().minorticks_on()
    plt.gca().grid()
    plt.gca().grid(which='minor', linestyle=":")
    plt.xlabel("L")
    plt.ylabel("Accuracy (%)")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig("Figure_1.pdf")
    print("Figure saved to Figure_1.pdf")

if __name__ == '__main__':
    main()
