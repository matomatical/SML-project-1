import sys
import importlib

import data
import models

def main():
    if len(sys.argv) <= 1:
        print("please specify a model name (e.g. models.baseline.random_handle)")
        sys.exit(1)
    module_name = sys.argv[1]
    hyper_parameters = {}
    for arg in sys.argv[2:]:
        hyper_parameters.update([arg.split("=")])

    module = importlib.import_module(module_name)
    model = module.Model(data.TRAIN, **hyper_parameters)

    # soz
    models.save(model, module_name + "-" + "-".join(f"{k}{v}" for k, v in hyper_parameters.items()))

if __name__ == '__main__':
    main()
