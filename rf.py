from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


def forester(X_train, X_test, y_train, y_test):
    parameters = {"estimators":500,"leaf nodes":7, "jobs":-1}
    rnd_clf=RandomForestClassifier(
        n_estimators=parameters["estimators"],
        max_leaf_nodes=parameters["leaf nodes"],
        n_jobs=parameters["jobs"]
        )
    rnd_clf.fit(X_train,y_train)
    y_pred_rf=rnd_clf.predict(X_test)
    report = classification_report(
        y_test,
        y_pred_rf, 
        output_dict=True
        )
    parameters = rnd_clf.get_params()
    # return report, parameters, hyperparameters (if there are any)
    return report, parameters


if __name__=="__main__":
    print('In progress')