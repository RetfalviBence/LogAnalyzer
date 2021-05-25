import matplotlib.pyplot as plt
from sklearn.metrics import auc

def roc_curve_plot(fpr, tpr, savePath=None, show=False):
    roc_auc = auc(fpr, tpr)
    plt.figure(dpi=150)
    plt.plot(fpr, tpr, lw=1, color='green', label=f'AUC = {roc_auc:.3f}')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.legend()

    if savePath is not None:
        plt.savefig(savePath)

    if show:
        plt.show()
    else:
        plt.close()

    return