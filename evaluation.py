import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import label_binarize

from sklearn.metrics import roc_auc_score


def get_performance(predictions, y_test, labels=[1, 0]):
    # Put your code
    accuracy = round(accuracy_score(y_test, predictions),2)
    [prec, rec,f1,support]=precision_recall_fscore_support(y_test, predictions, average='macro')
    precision = round(prec,2)
    recall = round(rec,2)
    f1_score = round(f1,2)
    
    report = classification_report(y_test, predictions,labels=labels)
    
    cm = confusion_matrix(y_test, predictions, labels=labels)
    cm_as_dataframe = pd.DataFrame(data=cm)
    
    print('Model Performance metrics:')
    print('-'*30)
    print('Accuracy:', accuracy)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1 Score:', f1_score)
    print('\nModel Classification report:')
    print('-'*30)
    print(report)
    print('\nPrediction Confusion Matrix:')
    print('-'*30)
    print(cm_as_dataframe)
    
    return accuracy, precision, recall, f1_score


def plot_roc(model, y_test, features):
    # Put your code
    predict_proba=model.predict_proba(features)
    # We want to keep the probability to obtain a "1", so we keep the second column
    one_probability=predict_proba[:,1]
    fpr, tpr, thresholds = metrics.roc_curve(y_test, one_probability)
    roc_auc = round(roc_auc_score(y_test, one_probability),3)

    plt.figure(figsize=(10, 5))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc})', linewidth=2.5)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

    return roc_auc