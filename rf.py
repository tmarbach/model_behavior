from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report


def forester(X_train, X_test, y_train, y_test, classes):
    parameters = {"estimators":500,"leaf nodes":8, "jobs":-1}
    rnd_clf=RandomForestClassifier(
        n_estimators=parameters["estimators"],
        max_leaf_nodes=parameters["leaf nodes"],
        n_jobs=parameters["jobs"]
        )
    rnd_clf.fit(X_train,y_train)
    y_pred_rf=rnd_clf.predict(X_test)
    #ytrainpred = cross_val_predict(svm_clf,X_train,y_train, cv=3)
   # conf_mx = confusion_matrix(y_test,y_pred_rf)
    report = classification_report(
        y_test,
        y_pred_rf, 
        target_names=classes,
        output_dict=True
        )
    parameters = rnd_clf.get_params()
    # return report, parameters, hyperparameters (if there are any)
    return report, parameters


if __name__=="__main__":
    print('In progress')