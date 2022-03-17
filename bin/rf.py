from pickle import TRUE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt



def forester(X_train, X_test, y_train, y_test, n_classes, classnames, output_fig_name):
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
        target_names = classnames,
        output_dict=True
        )
    recall_matrix = confusion_matrix(
        y_test,
        y_pred_rf,
        normalize = 'true'
        #labels=classnames
         )
    precision_matrix = confusion_matrix(
        y_test,
        y_pred_rf,
        normalize = 'pred'
        #labels=classnames
         )
    parameters = rnd_clf.get_params()
    plt.figure(figsize=(15,15))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred_rf,display_labels=classnames,normalize= 'true')
    plt.savefig('recall-' + str(output_fig_name))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred_rf,display_labels=classnames,normalize= 'pred')
    plt.savefig('precision-' + str(output_fig_name))

    return report, recall_matrix, precision_matrix, parameters


if __name__=="__main__":
    print('In progress')