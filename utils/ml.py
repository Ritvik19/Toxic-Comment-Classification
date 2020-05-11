from sklearn.metrics import accuracy_score, confusion_matrix, hamming_loss
from sklearn.model_selection import KFold
from sklearn.preprocessing import normalize
from sklearn.multiclass import OneVsRestClassifier

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

kf = KFold(n_splits=10, shuffle=True, random_state=101)

def train_model_one_vs_rest(model, vects, target, labels, **kwargs):
    model_performance = {
        'loss': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1 score': []
    }
    
    model = OneVsRestClassifier(model)

    for train_indices, test_indices in kf.split(vects, target):
        X_train = vects[train_indices]
        y_train = target[train_indices]

        X_test = vects[test_indices]
        y_test = target[test_indices]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        model_performance['loss'].append(hamming_loss(y_test, y_pred))
        model_performance['accuracy'].append(accuracy_score(y_test, y_pred))

    fig = plt.figure(figsize=(20, 18))

    ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=3)

    ax1.plot(model_performance['loss'], label='loss per iteration')
    ax1.plot(np.ones(10)*np.mean(model_performance['loss']), '--', label='mean loss')

    ax1.plot(model_performance['accuracy'], label='accuracy per iteration')
    ax1.plot(np.ones(10)*np.mean(model_performance['accuracy']), '--', label='mean accuracy')

    ax1.grid()
    ax1.legend()
    ax1.set_xlabel('fold')
    ax1.set_ylabel('value')
    ax1.set_title('Model Performance')

    cm = []
    
    
    for i, l in enumerate(labels):
        cm.append(normalize(confusion_matrix(y_test[:, i], y_pred[:, i]), axis=1, norm='l1'))
        ax2 = plt.subplot2grid((3, 3), (i//3+1, i%3))
        sns.heatmap(cm[-1], annot=True, square=True, ax=ax2, cmap='Blues')
        ax2.set_title(f'Confusion Matrix \'{l}\'')
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Actual')
    
    return model_performance, cm, model