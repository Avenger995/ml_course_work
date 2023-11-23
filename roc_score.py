import copy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def get_roc_auc_score(models, data_x, data_y, is_plotting=False):
    y_score = [i[0] for i in np.mean([model.predict(data_x, verbose=0) for model in models], axis=0)]
    y_true = copy.deepcopy(data_y.numpy())
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    if is_plotting:
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.show()
    return roc_auc
