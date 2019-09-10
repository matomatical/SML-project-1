import os
import glob
import json
from collections import defaultdict
from statistics import mean

HEADER = """<!DOCTYPE html>\n<html>\n<head>
<title>Peace, Love, Yoga</title>
<style>
main {max-width: 760px; margin: auto;}
table {border-collapse: collapse; border-spacing: 0; border: 1px solid #000;}
thead {background-color: #ddd; border-bottom: 1px solid;}
th, td {padding: 0.62em 1.5em;}
tr:nth-child(2n-1) td {background-color: #eee;}
</style>
</head>\n<body>\n<main>
<h1>Experimental results</h1>"""
TABLE_HEADER = """<h2>{filename}</h2>\n<table>
<thead>
 <th>Rank</th>
 <th>Hyper parameters</th>
 <th>Mean accuracy</th>
 <th>Folds tested</th>
</thead>"""
TABLE_ROW = """<tr>
 <td>{rank}</td>
 <td>{params}</td>
 <td>{accuracy:.2%}</td>
 <td>{folds}</td>
</tr>"""
TABLE_FOOTER = """</table>\n"""
FOOTER = """</main>\n</body>\n</html>\n"""


def main():
    with open("experiments/experiments-summary.html", 'w') as file:
        print(HEADER, file=file)
        for filename in glob.glob("experiments/*.jsonl"):
            print("processing", filename)
            print(TABLE_HEADER.format(filename=filename), file=file)
            
            # load the results
            results = []
            with open(filename) as infile:
                results = [json.loads(line) for line in infile]

            # collate and rank
            collated_results = defaultdict(list)
            for result in results:
                params = tuple(sorted(result['params'].items()))
                collated_results[params].append((result['fold'], result['accuracy']))
            scored_results = []
            for params, fold_accs in collated_results.items():
                folds, accs = zip(*fold_accs)
                scored_result = {
                    'params': ', '.join('='.join(map(str, p)) for p in params),
                    'accuracy': mean(accs), # TODO: Include variance?
                    'folds': ', '.join(map(str, folds))
                }
                scored_results.append(scored_result)
            ranked_results = sorted(scored_results, key=lambda r: -r['accuracy'])

            # generate table row
            for rank, result in enumerate(ranked_results, start=1):
                print(TABLE_ROW.format(rank=rank, **result), file=file)
            print(TABLE_FOOTER, file=file)
        
        print(FOOTER, file=file)
    filename = os.path.abspath("experiments/experiments-summary.html")
    print(f"saved result in file://{filename}")


if __name__ == '__main__':
    main()
