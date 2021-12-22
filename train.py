import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

from azureml.data.dataset_factory import TabularDatasetFactory
from azureml.core.run import Run

# Load data:
url_path = 'https://raw.githubusercontent.com/viyq/mle-OML-p2/7ad1d060ac2067d3cfa53b0aba28027fc56371c9/micropol-removal.csv'
dataset = TabularDatasetFactory.from_delimited_files(path=url_path)

df = dataset.to_pandas_dataframe()

y = df["rejection"]
X = df.drop(["Name", "ID","rejection"], axis="columns")

X_trans = OneHotEncoder(handle_unknown="ignore", sparse=False).fit_transform(X)
X_norm = MinMaxScaler().fit_transform(X_trans)
X_train, X_test, y_train, y_test = train_test_split(
    X_norm, y, test_size=0.2, random_state=7
)


run = Run.get_context()

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--hidden_layer_sizes', type=int, default=100, help="The ith element \
        represents the number of neurons in the ith hidden layer")
    parser.add_argument('--solver', type=str, default='lbfgs', help="The solver for weight optimization.")
    parser.add_argument('--activation', type=str, default='relu', help="Activation function for the hidden layer.")
    parser.add_argument('--alpha', type=float, default=0.0001, help="L2 penalty (regularization term) parameter")
    parser.add_argument('--tol', type=float, default=0.005, help="Tolerance for the early stopping")
    parser.add_argument('--max_iter', type=int, default=300, help="Maximum \
                        number of iterations to converge")

    args = parser.parse_args()

    run.log("Hidden layers:", np.int(args.hidden_layer_sizes))
    run.log("Solver:", np.str(args.solver))
    run.log("Activation:", np.str(args.activation))
    run.log("Alpha:", np.float(args.alpha))
    run.log("Tol:", np.float(args.tol))
    run.log("Max iterations:", np.int(args.max_iter))

    params = {
    "hidden_layer_sizes": args.hidden_layer_sizes,
    "solver": args.solver,
    "activation": args.activation,
    "alpha": args.alpha,
    "tol": args.tol,
    "max_iter": args.max_iter,
    }

    model = MLPRegressor(**params).fit(X_train, y_train)

    r2_score = model.score(X_test, y_test)
    run.log("r2_score", np.float(r2_score))
    
    model_name = "model_reg_" + str(args.solver) + "_" + str(args.activation) + "_" + str(r2_score) + ".pkl"
    filename = "./outputs/" + model_name
    
    joblib.dump(value=model, filename=filename)
    run.upload_file(name=model_name, path_or_stream=filename)
    run.complete()

if __name__ == '__main__':
    main()