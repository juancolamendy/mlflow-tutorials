import mlflow

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

import pandas as pd 

if __name__=="__main__":
    run_id = "d4e31c9d20f3443db20aedfba557f7fa"
        
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, random_state=42)
    print('type(X): ', type(X))
    print('shape(X): ', X.shape)
    print('type(y): ', type(y))
    print('shape(y): ', y.shape)
    X = pd.DataFrame(X, columns=["feature_{}".format(i) for i in range(10)])
    y = pd.DataFrame(y, columns=["target"])

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=43)
    print('type(y_test): ', type(y_test))
    print('shape(y_test): ', y_test.shape)
    print('type(X_test): ', type(X_test))
    print('shape(X_test): ', X_test.shape)

    # load model
    model_uri = f'runs:/{run_id}/random_forest_classifier'
    #rfc = mlflow.sklearn.load_model(model_uri=model_uri)
    rfc = mlflow.pyfunc.load_model(model_uri=model_uri)
    y_pred = rfc.predict(X_test)

    print('type(y_pred): ', type(y_pred))
    print('shape(y_pred): ', y_pred.shape)
    y_pred = pd.DataFrame(y_pred, columns=["prediction"])
    print('type(y_pred): ', type(y_pred))

    print('y_pred: ', y_pred[:5])
    print('y_test: ', y_test[:5])
    print('y_test.iloc[0]: ', y_test.iloc[0])