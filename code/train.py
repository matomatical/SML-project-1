import sys
import importlib

import data
import models

def main():
    if len(sys.argv) <= 1:
        print("please specify a model name (e.g. models.baseline.random_handle)")
        sys.exit(1)
    module_name = sys.argv[1]
    module = importlib.import_module(module_name)
    model = module.train(data.TRAIN)
    models.save(model, module_name)

if __name__ == '__main__':
    main()
