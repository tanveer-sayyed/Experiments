from pandas import DataFrame
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.model_selection import train_test_split as tts

best = 0
def optimise(X_train, X_test, y_train, y_test):
    param_space = {
    'max_depth': hp.choice('max_depth', range(1,20)),
    'max_features': hp.choice('max_features', range(1,5)),
    'n_estimators': hp.choice('n_estimators', range(100,500)),
    'criterion': hp.choice('criterion', ["gini", "entropy"])}

    def acc_model(params, X_train, y_train):
        clf = RandomForestClassifier(**params)
        clf.fit(X_train, y_train)
        return clf.score(X_test, y_test)
    
    def f(params):
        global best
        acc = acc_model(params, X_train, y_train)
        print(acc)
        if acc > best:
            best = acc
        return {'loss': -acc, 'status': STATUS_OK}
    
    trials = Trials()
    best = fmin(f, param_space, algo=tpe.suggest, max_evals=50, trials=trials)
    print ('best:')
    print (best)
    

if __name__ == '__main__':
    iris = load_iris()
    df = DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df['target'].replace(to_replace= [0,1,2], value= iris.target_names, inplace= True)
    df['target'] = df['target'].astype('object')
    X = df.drop(columns= 'target')
    y = df['target']
    X_train, X_test, y_train, y_test = tts(X, y, test_size= 0.3)
    
    optimise(X_train, X_test, y_train, y_test)