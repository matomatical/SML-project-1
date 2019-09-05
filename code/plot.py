import sys
import json
from statistics import mean
from collections import defaultdict
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
    for params, fold_accs in collated_results.items():
        L = [int(val) for key, val in params if key == "L"][0]
        desc = ', '.join('='.join(map(str, p)) for p in params if p[0] != "L")
        folds, accs = zip(*fold_accs)
        scored_results[desc][L] = mean(accs)

    plt.rc('axes', prop_cycle=cycler(color=list('rgbkcym')) * cycler(linestyle=['-','--','-.',':']))
    for desc, data in scored_results.items():
        plt.plot(*zip(*sorted(data.items())), marker='*', label=desc)
    plt.xlim(0, 1800)
    plt.xlabel("L")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
