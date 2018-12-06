#!/usr/bin/env python3

# Timothy Nguyen
# classification.py - use several algorithms to predict movie success
# algorithms: logistic regression, knn, decision tree, random forest

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier



df = pd.read_csv('moviesDb.csv')

predictors = ['budget','runtime','year','vote_average','vote_count','certification_US','genre','country']
x_dummy = pd.get_dummies(df.loc[:,predictors])

x = df[predictors].values
y = df['success'].values


############################################################
# print grid search results
def print_cv_results(gs, title):

    print(title)

    print(f'Best Score = {gs.best_score_:.4f}')
    print(f'Best Hyper-parameters = {gs.best_params_}')
    print()

    print('Test Scores:')
    test_means = gs.cv_results_['mean_test_score']
    test_stds = gs.cv_results_['std_test_score']
    for mean, std, params in zip(test_means, test_stds, gs.cv_results_['params']):
        print(f'{mean:.4f} (+/-{std:.3f}) for {params}')
    print()

    print('Training Scores:')
    train_means = gs.cv_results_['mean_train_score']
    train_stds = gs.cv_results_['std_train_score']
    for mean, std, params in zip(train_means, train_stds, gs.cv_results_['params']):
        print(f'{mean:.4f} (+/-{std:.3f}) for {params}')


# save grid search results to file
def save_cv_results(gs, title, fileName):
    with open(fileName, 'a') as f:

        print(title, file=f)

        print(f'Best Score = {gs.best_score_:.4f}', file=f)
        print(f'Best Hyper-parameters = {gs.best_params_}', file=f)
        print('', file=f)

        print('Test Scores:', file=f)
        test_means = gs.cv_results_['mean_test_score']
        test_stds = gs.cv_results_['std_test_score']
        for mean, std, params in zip(test_means, test_stds, gs.cv_results_['params']):
            print(f'{mean:.4f} (+/-{std:.3f}) for {params}', file=f)
        print('', file=f)

        print('Training Scores:', file=f)
        train_means = gs.cv_results_['mean_train_score']
        train_stds = gs.cv_results_['std_train_score']
        for mean, std, params in zip(train_means, train_stds, gs.cv_results_['params']):
            print(f'{mean:.4f} (+/-{std:.3f}) for {params}', file=f)


############################################################
# Logistic Regression
logReg = LogisticRegression()
c_list = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
param_grid = {'C': c_list,
              'solver': ['newton-cg', 'lbfgs', 'sag', 'saga']}

gs = GridSearchCV(estimator=logReg,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=10,
                  return_train_score=True)
gs = gs.fit(x_dummy, y)
print_cv_results(gs, 'Logistic Regression Accuracy')


# ############################################################
# # KNN
knn = KNeighborsClassifier()
k_list = list(range(1, 26, 2))
param_grid = [{'n_neighbors': k_list}]

gs = GridSearchCV(estimator=knn,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=10,
                  return_train_score=True)
gs = gs.fit(x_dummy, y)
print_cv_results(gs, 'KNN Accuracy')


test_means = gs.cv_results_['mean_test_score']
train_means = gs.cv_results_['mean_train_score']

plt.plot(k_list, test_means, marker='o', label='Test')
plt.plot(k_list, train_means, marker='o', label='Train')
plt.xticks(k_list)

plt.title('Movie Success Prediction: KNN')
plt.ylabel('Accuracy')
plt.xlabel('Number of Neighbors')
plt.legend()
plt.show()


############################################################
# SVM
# This takes too much time to run
# svm = SVC()
#
# param_range = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
# param_grid = [{'kernel': ['linear', 'rbf', 'poly'],
#                'C': param_range}]
# gs = GridSearchCV(estimator=svm,
#                   param_grid=param_grid,
#                   scoring='accuracy',
#                   cv=10,
#                   n_jobs=-1)
# gs = gs.fit(x_dummy, y)
# print_cv_results(gs, 'SVC Accuracy')


# ############################################################
# Decision Tree
criterions = ['gini', 'entropy']
colors = ['red', 'blue']
depth_list = list(range(1,11))

for i in range(len(criterions)):
    tree = DecisionTreeClassifier(criterion=criterions[i])
    param_grid = [{'max_depth': depth_list}]
    gs = GridSearchCV(estimator=tree,
                      param_grid=param_grid,
                      scoring='accuracy',
                      cv=10)
    gs = gs.fit(x_dummy, y)
    print_cv_results(gs, 'Decision Tree Regression Accuracy')

    test_means = gs.cv_results_['mean_test_score']
    train_means = gs.cv_results_['mean_train_score']

    plt.plot(depth_list, test_means, marker='o', label=f'{criterions[i]} Test Mean',
                color=colors[i])
    plt.plot(depth_list, train_means, marker='o', label=f'{criterions[i]} Train Mean',
                linestyle='dashed', color=colors[i])

plt.xticks(depth_list)
plt.title(f'Movie Success Prediction: Decision Tree')
plt.ylabel('Accuracy')
plt.xlabel('Max Tree Depth')
plt.legend()
plt.show()


# # ############################################################
# # Random Forest
# get results for random forest
forest = RandomForestClassifier()
criterions = ['gini', 'entropy']
n_list = list(range(1, 11))
param_grid = [{'n_estimators': n_list,
                'max_depth': n_list,
                'criterion': criterions}]
gs = GridSearchCV(estimator=forest,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=10)
gs = gs.fit(x_dummy, y)
save_cv_results(gs, 'Random Forest Accuracy', 'rand-forest.txt')


# print line graph of random forest where max_depth=8
criterions = ['gini', 'entropy']
colors = ['red', 'blue']
n_list = list(range(1, 11))
for i in range(len(criterions)):
    forest = RandomForestClassifier(criterion=criterions[i], max_depth=8)
    param_grid = [{'n_estimators': n_list}]
    gs = GridSearchCV(estimator=forest,
                      param_grid=param_grid,
                      scoring='accuracy',
                      cv=10)
    gs = gs.fit(x_dummy, y)
    print_cv_results(gs, 'Random Forest Accuracy')

    test_means = gs.cv_results_['mean_test_score']
    train_means = gs.cv_results_['mean_train_score']

    plt.plot(n_list, test_means, marker='o', label=f'{criterions[i]} Test Mean',
                color=colors[i])
    plt.plot(n_list, train_means, marker='o', label=f'{criterions[i]} Train Mean',
                linestyle='dotted', color=colors[i])

plt.xticks(n_list)
plt.title(f'Movie Success Prediction: Random Forest, max_depth=8')
plt.ylabel('Accuracy')
plt.xlabel('Number of Trees')
plt.legend()
plt.show()
