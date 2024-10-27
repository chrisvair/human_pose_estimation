from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
import argparse
from sklearn.metrics import precision_score, recall_score
import numpy as np

"""
This file will run through a number of scikit learn models on the the training data
in training.csv.  This training data was collected through running:
"""

model_name = 'rdm'

def get_data(file_name):
    """
    read data.csv and return the X,y as series
    :return: X - DataFrame of features
             y - Series of labels
    """
    df = pd.read_csv(f'{file_name}', header=None)
    X = df.loc[:, 1:]
    y = df.loc[:, 0]
    classes = []
    if y.dtype == object:
        le = LabelEncoder()
        y = le.fit_transform(y)
        classes = le.classes_
    return X, y, classes

def train_model(model, X, y, name=None, param_grid=None):
    """
        Trains a machine learning model using cross-validation and optionally performs hyperparameter tuning with grid search.

        Parameters:
        model : estimator object
            A scikit-learn estimator instance (e.g., a classifier or regressor) to be trained.

        X : DataFrame
            The input data to fit.

        y : Series
            The target variable to predict.ss

        name : str, optional
            The name of the model, used for printing information.

        param_grid : dict or list of dictionaries, optional
            Dictionary with parameters names (str) as keys and lists of parameter settings to try as values,
            or a list of such dictionaries, in case of hyperparameter tuning.

        Returns:
        tuple
            A tuple containing the best score (float), best parameters (dict or None), and the best model (estimator object).
    """
    if name:
        print(f"Training: {name}")
    if param_grid:
        # grid search with 5-fold cross-validation to find the best hyperparameters
        grid = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
        grid.fit(X, y)
        _best_model = grid.best_estimator_
        _best_params = grid.best_params_
        _best_score = grid.best_score_
    else:
        # 5-fold cross-validation to evaluate the model's performance
        cv_scores = cross_val_score(model, X, y, cv=5)
        model.fit(X, y)
        _best_model = model
        _best_params = param_grid
        _best_score = cv_scores.mean()
    # Calculate precision and recall
    #y_pred = cross_val_predict(model, X, y, cv=5)
    #precision = precision_score(y, y_pred, average='weighted')
    #recall = recall_score(y, y_pred, average='weighted')
    #print("Precision:", precision)
    #print("Recall:", recall)
    print(_best_score)
    print(_best_params)
    return _best_score, _best_params, _best_model

# Instances of various machine learning models :
# LogisticRegression, DecisionTree, SVC, MultinomialNB, KNN, Linear, RandomForestClassifier, GradientBoostingClassifier, MLP
def create_logistic_regression_model():
    return LogisticRegression()

def create_decision_tree():
    return DecisionTreeClassifier()

def create_svc():
    return SVC(kernel='linear', C=1)

def create_rbf_svc():
    return SVC(kernel='rbf')

def create_mnb():
    return GaussianNB()

def create_knn():
    return KNeighborsClassifier()

def create_linear():
    return LinearRegression()

def create_mlp():
    return MLPClassifier()

# Ensemble model combining various classifiers
def create_voting_classifier():
    logreg = LogisticRegression()
    tree = DecisionTreeClassifier()
    svc = SVC(probability=True)
    knn = KNeighborsClassifier()
    rf = RandomForestClassifier()
    gb = GradientBoostingClassifier()
    mlp = MLPClassifier()
    voting_clf = VotingClassifier(
        estimators=[
            ('logreg', logreg),
            ('tree', tree),
            ('svc', svc),
            ('knn', knn),
            ('rf', rf),
            ('gb', gb),
            ('mlp', mlp)
        ],
        voting='soft'
    )
    return voting_clf

# Try various machine learning models, hyperparameters, and configurations to find the best model for the given data.
def find_best_model(X, y):
    #list of dictionaries, each containing information about a specific model to be trained.
    # It includes the model type, parameter grid for hyperparameter tuning, name of the model, and whether to skip training it.
    models = [
        {
            'model': make_pipeline(StandardScaler(), create_logistic_regression_model()),
            'params_grid': None,
            'name': 'LogisticRegression',
            'skip': False
        },
        # 0.9983935742971888
        {
            'model': make_pipeline(StandardScaler(), create_decision_tree()),
            'params_grid': {
                'decisiontreeclassifier__criterion': ['gini', 'entropy'],
                'decisiontreeclassifier__max_depth': [2, 3, 4, 5],
                'decisiontreeclassifier__min_samples_split': [2, 3]
            },
            'name': 'DecisionTree',
            'skip': False
        },
        # 0.8605616012436844
        # 0.8848199248607334 without hyperparameters tuning
        {
            'model': make_pipeline(StandardScaler(), create_svc()),
            'params_grid': {
                'svc__kernel': ['linear', 'rbf', 'poly'],
                'svc__C': [1, 10, 100, 1000],
                'svc__gamma': ['scale', 'auto']
            },
            # 'params_grid': None, # nothing changes
            'name': 'SVC',
            'skip': False
        },
        # 0.9951677678455759
        {
            'model': make_pipeline(StandardScaler(), create_mnb()),
            'params_grid': None,
            'name': 'MultinomialNB',
            'skip': False
        },
        # 0.7946981474284234
        {
            'model': make_pipeline(StandardScaler(), create_knn()),
            'params_grid': {
                'kneighborsclassifier__n_neighbors': list(range(1, 10)),
                'kneighborsclassifier__weights': ['uniform', 'distance']
            },
            'name': 'KNN',
            'skip': False
        },
        # 0.9501360279828994 with hyperparameter tuning
        # 0.9388457054022542
        {
            'model': make_pipeline(StandardScaler(), RandomForestClassifier()),
            'params_grid': None,
            'name': 'RandomForestClassifier',
            'skip': False
        },
        # 0.9243036662780153
        {
            'model': make_pipeline(StandardScaler(), GradientBoostingClassifier()),
            'params_grid': {
                'gradientboostingclassifier__n_estimators': [100, 200],
                'gradientboostingclassifier__learning_rate': [0.01, 0.1, 1.0],
                'gradientboostingclassifier__max_depth': [3, 4, 5]
            },
            'name': 'GradientBoostingClassifier',
            'skip': False
        },
        # 0.954048451872004 with hyperparameter tuning
        # 0.9411840912035239
        {
            'model': make_pipeline(StandardScaler(), create_mlp()),
            'params_grid': None,
            'name': 'MLP',
            'skip': False
        }
        # 0.9871194455240317
    ]
    best_model = None
    best_params = None
    best_score = -1
    for model in models:
        if not model['skip']:
            score, params, best = train_model(model['model'], X, y, name=model['name'], param_grid=model['params_grid'])
            if score > best_score:
                best_params = params
                best_model = best
                best_score = score
    return best_model, best_params, best_score

'''
python model_training.py --file-name ymca_training.csv --model-name ymca_pose_model

'''

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--training-data", type=str, required=False, default='../data/robot.csv',
                    help="name of the training data file")
    ap.add_argument("--model-name", type=str, required=False, default=f'{model_name}',
                    help=f"name of the saved pickled model [no suffix]. Default: {model_name}.pkl")
    args = vars(ap.parse_args())

    model_name = args['model_name']

    X, y, classes = get_data(args['training_data'])

    best_model, best_params, best_score = find_best_model(X, y)

    print("*******  Best Model and Parameters  *********")
    print(best_model)
    print(best_params)
    print(best_score)
    with open(f'{model_name}_metadata.txt', 'w') as f:
        f.write(f'{best_model}\n')
        f.write(f'{best_params}\n')
        f.write(f'{best_score}\n')

    with open(f'{model_name}_classes.txt', 'w') as f:
        f.write(f"{classes}")

    joblib.dump(best_model, f"{model_name}.pkl")

    print(f"Done saving model to best model:  {model_name}.pkl")