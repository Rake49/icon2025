import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import KFold, RepeatedKFold, learning_curve, train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
import json

# Funzione che mostra la curva di apprendimento per ogni modello
def plot_learning_curves(model, X, y, differentialColumn, model_name):
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=10, scoring='neg_log_loss')

    # Calcola gli errori su addestramento e test
    train_errors = train_scores
    test_errors = test_scores

    # Calcola la deviazione standard e la varianza degli errori su addestramento e test
    train_errors_std = np.std(train_errors, axis=1)
    test_errors_std = np.std(test_errors, axis=1)
    train_errors_var = np.var(train_errors, axis=1)
    test_errors_var = np.var(test_errors, axis=1)

    # Stampa i valori numerici della deviazione standard e della varianza
    log_message = (f"{model_name} - Train Error Std: {train_errors_std[-1]}, "
                   f"Test Error Std: {test_errors_std[-1]}, Train Error Var: {train_errors_var[-1]}, "
                   f"Test Error Var: {test_errors_var[-1]}")
    # Log the message to a file
    with open('learning_curve_log.txt', 'a') as log_file:
        log_file.write(log_message + '\n')

    # Calcola gli errori medi su addestramento e test
    mean_train_errors = np.mean(train_scores, axis=1) * -1
    mean_test_errors = np.mean(test_scores, axis=1) * -1

    # Visualizza la curva di apprendimento
    plt.figure(figsize=(16, 10))
    plt.plot(train_sizes, mean_train_errors,
             label='Errore di training', color='green')
    plt.plot(train_sizes, mean_test_errors,
             label='Errore di testing', color='red')
    plt.title(f'Curva di apprendimento per {model_name}')
    plt.xlabel('Dimensione del training set')
    plt.ylabel('Errore')
    plt.legend()
    plt.show()

# Funzione che restituisce i migliori iperparametri per ogni modello
def returnBestHyperparametres(dataset, differentialColumn):
    X = dataset.drop(differentialColumn, axis=1).to_numpy()
    y = dataset[differentialColumn].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#creazione modelli

    decTree = DecisionTreeClassifier()
    randFor = RandomForestClassifier()
    neurNet = MLPClassifier()
    catBoost = CatBoostClassifier()

#iperparametri modelli
    decTreeParameters = {
        'DecisionTree__criterion': ('gini', 'log_loss'),
        'DecisionTree__splitter': ['best', 'random'],
        'DecisionTree__min_samples_split': [2, 5], 
        'DecisionTree__min_samples_leaf': [1, 3], 
        'DecisionTree__max_features': [None, 'sqrt']
    }
    randForParameters = {
        'RandomForest__n_estimators': [50, 100, 150],
        'RandomForest__min_samples_split': [2, 5], 
        'RandomForest__min_samples_leaf': [1, 3], 
        'RandomForest__max_features': [None, 'sqrt']
    }
    neurNetParameters = {
        'neurNet__hidden_layer_sizes': [(20,), (40,), (20, 10), (40,20)],
        'neurNet__activation': ['logistic', 'relu'],
        'neurNet__solver': ['sgd', 'adam'],
        'neurNet__alpha': [0.0001, 0.05],
        'neurNet__learning_rate': ['constant', 'adaptive'],
        'neurNet__max_iter': [2000]
    }
    catBoostParameters = {
        'catBoost__iterations': [100, 200, 300],
        'catBoost__depth': [6, 7, 8],
        'catBoost__learning_rate': [0.01, 0.05, 0.1],
        'catBoost__loss_function': ['Logloss']
    }

#ricerca parametrizzata
    gridSearchCV_catBoost = GridSearchCV(
        Pipeline([('catBoost', catBoost)]), catBoostParameters, cv=5)
    gridSearchCV_decTree = GridSearchCV(
        Pipeline([('DecisionTree', decTree)]), param_grid=decTreeParameters, cv=5)
    gridSearchCV_randFor = GridSearchCV(
        Pipeline([('RandomForest', randFor)]), param_grid=randForParameters, cv=5)
    gridSearchCV_neurNet = GridSearchCV(
        Pipeline([('neurNet', neurNet)]), param_grid=neurNetParameters, cv=5)

    gridSearchCV_decTree.fit(X_train, y_train)
    gridSearchCV_randFor.fit(X_train, y_train)
    gridSearchCV_neurNet.fit(X_train, y_train)
    gridSearchCV_catBoost.fit(X_train, y_train)

    bestParameters = {
        'DecisionTree__criterion': gridSearchCV_decTree.best_params_['DecisionTree__criterion'],
        'DecisionTree__splitter': gridSearchCV_decTree.best_params_['DecisionTree__splitter'],
        'decTreeMin_samples_split': gridSearchCV_decTree.best_params_['DecisionTree__min_samples_split'],
        'decTreeMin_samples_leaf': gridSearchCV_decTree.best_params_['DecisionTree__min_samples_leaf'],
        'decTreeMax_features': gridSearchCV_decTree.best_params_['DecisionTree__max_features'],

        'n_estimators': gridSearchCV_randFor.best_params_['RandomForest__n_estimators'],
        'randForMin_samples_split': gridSearchCV_randFor.best_params_['RandomForest__min_samples_split'],
        'randForMin_samples_leaf': gridSearchCV_randFor.best_params_['RandomForest__min_samples_leaf'],
        'randForMax_features': gridSearchCV_randFor.best_params_['RandomForest__max_features'],

        'neurNet__hidden_layer_sizes': gridSearchCV_neurNet.best_params_['neurNet__hidden_layer_sizes'],
        'neurNet__activation': gridSearchCV_neurNet.best_params_['neurNet__activation'],
        'neurNet__solver': gridSearchCV_neurNet.best_params_['neurNet__solver'],
        'neurNet__alpha': gridSearchCV_neurNet.best_params_['neurNet__alpha'],
        'neurNet__learning_rate': gridSearchCV_neurNet.best_params_['neurNet__learning_rate'],
        'neurNet__max_iter': gridSearchCV_neurNet.best_params_['neurNet__max_iter'],

        'catBoost__iterations': gridSearchCV_catBoost.best_params_['catBoost__iterations'],
        'catBoost__depth': gridSearchCV_catBoost.best_params_['catBoost__depth'],
        'catBoost__learning_rate': gridSearchCV_catBoost.best_params_['catBoost__learning_rate'],
        'catBoost__loss_function': gridSearchCV_catBoost.best_params_['catBoost__loss_function']
    }
    return bestParameters

# Funzione che esegue il training del modello mediante cross validation
def trainModelKFold(dataSet, differentialColumn):
    model = {
        'DecisionTree': {
            'accuracy': [],
            'precision': [],
            'f1': [],
            'recall': [],
        },
        'RandomForest': {
            'accuracy': [],
            'precision': [],
            'f1': [],
            'recall': [],
        },
        'neurNet': {
            'accuracy': [],
            'precision': [],
            'f1': [],
            'recall': []
        },
        'catBoost': {
            'accuracy': [],
            'precision': [],
            'f1': [],
            'recall': []
        }
    }
    validation_set = dataSet.sample(frac = 0.25)
    test_set = dataSet.drop(validation_set.index)
    bestParameters = returnBestHyperparametres(validation_set, differentialColumn)

    # Log best parameters to a file
    with open('best_parameters.json', 'w') as file:
        json.dump(bestParameters, file, indent=4)
    X = test_set.drop(differentialColumn, axis=1).to_numpy()
    y = test_set[differentialColumn].to_numpy()

    decTree = DecisionTreeClassifier(criterion=bestParameters['DecisionTree__criterion'],
                                splitter=bestParameters['DecisionTree__splitter'],
                                min_samples_split=bestParameters['decTreeMin_samples_split'],
                                min_samples_leaf=bestParameters['decTreeMin_samples_leaf'],
                                max_features=bestParameters['decTreeMax_features'])
    randFor = RandomForestClassifier(n_estimators=bestParameters['n_estimators'],
                                # splitter='randForSplitter',
                                min_samples_split=bestParameters['randForMin_samples_split'],
                                min_samples_leaf=bestParameters['randForMin_samples_leaf'],
                                max_features=bestParameters['randForMax_features'])
    neurNet = MLPClassifier(hidden_layer_sizes = bestParameters['neurNet__hidden_layer_sizes'],
                                activation = bestParameters['neurNet__activation'],
                                solver = bestParameters['neurNet__solver'],
                                alpha = bestParameters['neurNet__alpha'],
                                learning_rate = bestParameters['neurNet__learning_rate'],
                                max_iter = bestParameters['neurNet__max_iter']
    )
    catBoost = CatBoostClassifier(iterations=bestParameters['catBoost__iterations'],
                                depth=bestParameters['catBoost__depth'],
                                learning_rate=bestParameters['catBoost__learning_rate'],
                                loss_function=bestParameters['catBoost__loss_function'])

    cv = RepeatedKFold(n_splits=5, n_repeats=5)

    scoring_metrics = ['accuracy', 'precision', 'f1', 'recall']

    results_decTree = {}
    results_randFor = {}
    results_neurNet = {}
    results_catBoost = {}

    for metric in scoring_metrics:
        scores_decTree = cross_val_score(decTree, X, y, scoring=metric, cv=cv)
        scores_randFor = cross_val_score(randFor, X, y, scoring=metric, cv=cv)
        scores_neurNet = cross_val_score(neurNet, X, y, scoring=metric, cv=cv)
        scores_catBoost = cross_val_score(catBoost, X, y, scoring=metric, cv=cv)

        results_decTree[metric] = scores_decTree
        results_randFor[metric] = scores_randFor
        results_neurNet[metric] = scores_neurNet
        results_catBoost[metric] = scores_catBoost

    model['DecisionTree']['accuracy'] = (results_decTree['accuracy'])
    model['DecisionTree']['precision'] = (results_decTree['precision'])
    model['DecisionTree']['recall'] = (results_decTree['recall'])
    model['DecisionTree']['f1'] = (results_decTree['f1'])
    model['RandomForest']['accuracy'] = (results_randFor['accuracy'])
    model['RandomForest']['precision'] = (results_randFor['precision'])
    model['RandomForest']['recall'] = (results_randFor['recall'])
    model['RandomForest']['f1'] = (results_randFor['f1'])
    model['neurNet']['accuracy'] = (results_neurNet['accuracy'])
    model['neurNet']['precision'] = (results_neurNet['precision'])
    model['neurNet']['recall'] = (results_neurNet['recall'])
    model['neurNet']['f1'] = (results_neurNet['f1'])
    model['catBoost']['accuracy'] = (results_catBoost['accuracy'])
    model['catBoost']['precision'] = (results_catBoost['precision'])
    model['catBoost']['recall'] = (results_catBoost['recall'])
    model['catBoost']['f1'] = (results_catBoost['f1'])

    plot_learning_curves(decTree, X, y, differentialColumn, 'DecisionTree')
    plot_learning_curves(randFor, X, y, differentialColumn, 'RandomForest')
    plot_learning_curves(neurNet, X, y, differentialColumn, 'neurNet')
    plot_learning_curves(catBoost, X, y, differentialColumn, 'catBoost')

    visualizeMetricsGraphs(model)
    return model

# Funzione che visualizza i grafici delle metriche per ogni modello
def visualizeMetricsGraphs(model):
    models = list(model.keys())
    
    # Creazione di un array numpy per ogni metrica

    precision = np.array([model[clf]['precision'] for clf in models])
    accuracy = np.array([model[clf]['accuracy'] for clf in models])
    f1 = np.array([model[clf]['f1'] for clf in models])
    recall = np.array([model[clf]['recall'] for clf in models])

    # Calcolo delle medie per ogni modello e metrica
    mean_precision = np.mean(precision, axis=1)
    mean_accuracy = np.mean(accuracy, axis=1)
    mean_f1 = np.mean(f1, axis=1)
    mean_recall = np.mean(recall, axis=1)

    yint = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    # Creazione del grafico a barre per Precision
    plt.figure(figsize=(12, 6))
    bar_width = 0.4
    index = np.arange(len(models))
    plt.bar(index, mean_precision, bar_width, label='precision')
    plt.xlabel('Modelli')
    plt.ylabel('Precision media')
    plt.yticks(yint)
    plt.title('Precision media per ogni modello')
    plt.xticks(index, models)
    plt.legend()
    plt.show()

    # Creazione del grafico a barre per Accuracy
    plt.figure(figsize=(12, 6))
    bar_width = 0.4
    index = np.arange(len(models))
    plt.bar(index, mean_accuracy, bar_width, label='accuracy')
    plt.xlabel('Modelli')
    plt.ylabel('Accuracy media')
    plt.title('Accuracy media per ogni modello')
    plt.xticks(index, models)
    plt.yticks(yint)
    plt.legend()
    plt.show()

    # Creazione del grafico a barre per F1
    plt.figure(figsize=(12, 6))
    bar_width = 0.4
    index = np.arange(len(models))
    plt.bar(index, mean_f1, bar_width, label='f1')
    plt.xlabel('Modelli')
    plt.ylabel('F1 media')
    plt.title('F1 media per ogni modello')
    plt.xticks(index, models)
    plt.yticks(yint)
    plt.legend()
    plt.show()

    # Creazione del grafico a barre per Recall
    plt.figure(figsize=(12, 6))
    bar_width = 0.4
    index = np.arange(len(models))
    plt.bar(index, mean_recall, bar_width, label='recall')
    plt.xlabel('Modelli')
    plt.ylabel('Recall media')
    plt.title('Recall media per ogni modello')
    plt.xticks(index, models)
    plt.yticks(yint)
    plt.legend()
    plt.show()