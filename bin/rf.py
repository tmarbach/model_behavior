from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


def forester(X_train, X_test, y_train, y_test, n_classes):
    #parameters = {"estimators":1000,"leaf nodes":n_classes, "jobs":-1}
    rnd_clf=RandomForestClassifier(
        n_estimators=1000,
        max_leaf_nodes=n_classes,
        n_jobs=-1
        )
    rnd_clf.fit(X_train,y_train)
    y_pred_rf=rnd_clf.predict(X_test)
    report = classification_report(
        y_test,
        y_pred_rf, 
        output_dict=True
        )
    parameters = rnd_clf.get_params()
    return report, parameters


if __name__=="__main__":
    print('In progress')