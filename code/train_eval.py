import sys
import importlib

from tqdm import tqdm

import data
import eval
import models

data.load_train()
data.load_devel()

def main():
    if len(sys.argv) <= 1:
        print("please specify a model name (e.g. models.baseline.random_handle)")
        sys.exit(1)
    module_name = sys.argv[1]
    hyper_parameters = models.parse_hyper_parameters(sys.argv[2:])
    model_class = importlib.import_module(module_name).Model
    print("Model:", module_name, hyper_parameters)

    print("Training...")
    model = model_class(tqdm(data.TRAIN, dynamic_ncols=True), **hyper_parameters)
    print("Evaluating...")
    accuracy, correct, tests = eval.evaluate(model, tqdm(data.DEVEL, dynamic_ncols=True))
    print(f"Label accuracy: {correct}/{tests} ({accuracy:%})")


if __name__ == '__main__':
    main()
