import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model

from sklearn import datasets, ensemble
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def show_Ordinary_Least_Squares_vs_Ridge_Regression_Variance(X, y, test_size=0.33, random_state=42):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    
    classifiers = dict(ols=linear_model.LinearRegression(),
                   ridge=linear_model.Ridge(alpha=.1))
    

    for name, clf in classifiers.items():
        fig, ax = plt.subplots(figsize=(4, 3))

        for _ in range(6):
            this_X = .1 * np.random.normal(size=(X_train.shape[0], 1)) + X_train
            clf.fit(this_X, y_train)

            ax.plot(X_test, clf.predict(X_test), color='gray')
            ax.scatter(this_X, y_train, s=3, c='gray', marker='o', zorder=10)

        clf.fit(X_train, y_train)
        ax.plot(X_test, clf.predict(X_test), linewidth=2, color='blue')
        ax.scatter(X_train, y_train, s=30, c='red', marker='+', zorder=10)

        ax.set_title(name)
        # ax.set_xlim(0, 2)
        # ax.set_ylim((0, 1.6))
        ax.set_xlabel('X')
        ax.set_ylabel('y')

        fig.tight_layout()

    plt.show()
    pass


# -------------------------------------------------------------------------------------------
# Gradient Boosting regression Simulation
# https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regression.html#sphx-glr-auto-examples-ensemble-plot-gradient-boosting-regression-py
# -------------------------------------------------------------------------------------------

def plot_training_deviance(params, X_test, y_test):
    test_score = np.zeros((params['n_estimators'],), dtype=np.float64)
    for i, y_pred in enumerate(reg.staged_predict(X_test)):
        test_score[i] = reg.loss_(y_test, y_pred)

    fig = plt.figure(figsize=(6, 6))
    plt.subplot(1, 1, 1)
    plt.title('Deviance')
    plt.plot(np.arange(params['n_estimators']) + 1, reg.train_score_, 'b-',
         label='Training Set Deviance')
    plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
         label='Test Set Deviance')
    plt.legend(loc='upper right')
    plt.xlabel('Boosting Iterations')
    plt.ylabel('Deviance')
    fig.tight_layout()
    plt.show()
    pass

def exmaple_gradient_boosting_regression(X, y, test_size=0.1, random_state=13, verbose=0):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    params = {'n_estimators': 500,
          'max_depth': 4,
          'min_samples_split': 5,
          'learning_rate': 0.01,
          'loss': 'ls'}
    reg = ensemble.GradientBoostingRegressor(**params)
    reg.fit(X_train, y_train)
    
    mse = mean_squared_error(y_test, reg.predict(X_test))
    if verbose == 1:
        print("The mean squared error (MSE) on test set: {:.4f}".format(mse))
    
    plot_training_deviance(params, X_test, y_test)
    pass