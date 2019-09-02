import data
import eval
from models.baseline.simple_ngram import Model
import sklearn.model_selection as model_selection
import numpy as np

# keys: hyper parameters, values: list of strings (values of hyperparameters)
GRID ={
  "L": [200, 500, 1000],
  "n": [2, 3, 4, 5]
}
N_SPLITS = 8

DATA = np.array(data.TRAIN)

# generate 
kfold = model_selection.KFold(n_splits=N_SPLITS, random_state=51)

for params in model_selection.ParameterGrid(GRID):
  print("Trying", params)
  accuracies = []
  for i, (train_index, test_index) in enumerate(kfold.split(DATA), start=1):
    print("Fold", i)
    # train and evaluate a model
    print("training...")
    train_data = DATA[train_index]
    model = Model(train_data, **params)
    print("evaluating...")
    accuracy, *_ = eval.evaluate(model, DATA[test_index])
    accuracies.append(accuracy)
    print(f"accuracy: {accuracy:.2%}")
  avg_accuracy = sum(accuracies)/len(accuracies)
  print(f"Finished testing {params}. Mean accuracy from {N_SPLITS} folds: {avg_accuracy:.2%}")
