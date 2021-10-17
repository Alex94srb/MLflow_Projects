import os
import tempfile
import click
import warnings
import sys

import numpy as np
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from matplotlib import pyplot as plt



def get_temporary_directory_path(prefix, suffix):
    """
    Get a temporary directory and files for artifacts
    :param prefix: name of the file
    :param suffix: .csv, .txt, .png etc
    :return: object to tempfile.
    """

    temp = tempfile.NamedTemporaryFile(prefix=prefix, suffix=suffix)
    return temp

def plot_graphs(x_data, y_data, x_label, y_label, title):
    """
    Use the Mathplot lib to plot data points provide and respective x-axis and y-axis labels
    :param x_data: Data for x-axis
    :param y_data: Data for y-axis
    :param x_label: Label for x-axis
    :param y_label: Label FOR Y-axis
    :param title: Title for the plot
    :return: return tuple (fig, ax)
    """

    plt.clf()

    fig, ax = plt.subplots()
    ax.plot(x_data, y_data)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    return (fig, ax)


# @click.command()
# @click.option("--params", type=dict, default={'n_estimators': 50, 'max_depth': 6}, help="Dictionary with parameters for Random Forest Regressor")
# @click.option("--r-name", default="Lab-2:RF Petrol Regression Experiment - Projects", type=str, help="Name of the MLflow run")   
def train_random_forest_reg(params):
    """
    This method trains, computes metrics, and logs all metrics, parameters,
    and artifacts for the current run using the MLflow APIs
    :param df: pandas dataFrame
    :param r_name: Name of the run as logged by MLflow
    :return: MLflow Tuple (ExperimentID, runID)
    """
    warnings.filterwarnings("ignore")
    

        # Read the wine-quality csv file (make sure you're running this from the root of MLflow!)
    petrol_cons = os.path.join(os.path.dirname(os.path.abspath(__file__)), "petrol_consumption.csv")
    df = pd.read_csv(petrol_cons)

    # <------------------- MLflow ------------------->
    with mlflow.start_run() as run:
        # define the random forest regressor model
        rf = RandomForestRegressor(**params)

        # get current run and experiment id
        runID = run.info.run_uuid
        experimentID = run.info.experiment_id
        
        # extract all feature independent attributes
        X = df.iloc[:, 0:4].values
        # extract all the values of last columns, dependent variables,
        # which is what we want to predict as our values, the petrol consumption
        y = df.iloc[:, 4].values

        # create train and test data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        # Feature Scaling, though for RF is not necessary.
        # z = (X - u)/ s, where u is the mean, s the standard deviation
        # get the handle to the transformer
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        # train and predict
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        # Log model and params using the MLflow APIs
        rf.params = rf.get_params()
        mlflow.sklearn.log_model(rf, "random-forest-reg-model")
        mlflow.log_params(rf.params)

        # compute  regression evaluation metrics 
        mae = metrics.mean_absolute_error(y_test, y_pred)
        mse = metrics.mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = metrics.r2_score(y_test, y_pred)

        # Log metrics
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        # update global class instance variable with values
        rf.rmse = []
        rf.estimators = []
        rf.rmse.append(rmse)
        rf.estimators.append(rf.params["n_estimators"])

        # plot graphs and save as artifacts
        (fig, ax) = plot_graphs(rf.estimators, rmse, "Random Forest Estimators", "Root Mean Square", "Root Mean Square vs Estimators")

        # create temporary artifact file name and log artifact
        temp_file_name = get_temporary_directory_path("rmse_estimators-", ".png")
        temp_name = temp_file_name.name
        try:
            fig.savefig(temp_name)
            mlflow.log_artifact(temp_name, "rmse_estimators_plots")
        finally:
            temp_file_name.close()  # Delete the temp file

        # print some data
        print("-" * 100)
        print("Inside MLflow Run with run_id {} and experiment_id {}".format(runID, experimentID))
        print("Estimator trees        :", rf.params["n_estimators"])
        print('Mean Absolute Error    :', mae)
        print('Mean Squared Error     :', mse)
        print('Root Mean Squared Error:', rmse)
        print('R2                     :', r2)
        
        return (experimentID, runID)


if __name__ == "__main__":
    params = dict(sys.argv[1]) if len(sys.sys.argv) > 1 else {'n_estimators': 50, 'max_depth': 6}

    train_random_forest_reg(params=params)