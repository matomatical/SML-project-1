import sys
import importlib

from tqdm import tqdm

import data; data.load_train()
import models

def main():
    # interpret command line arguments
    if len(sys.argv) <= 1:
        print("please specify a model name (e.g. models.baseline.random_handle)")
        sys.exit(1)
    module_name = sys.argv[1]
    hyper_parameters = models.parse_hyper_parameters(sys.argv[2:])

    # training model
    module = importlib.import_module(module_name)
    print(f"Training {module_name}.Model with hyperparameters {hyper_parameters}")
    model = module.Model(tqdm(data.TRAIN), **hyper_parameters)
    print("Training done!")
    
    models.save(model, module_name, hyper_parameters)

if __name__ == '__main__':
    main()
