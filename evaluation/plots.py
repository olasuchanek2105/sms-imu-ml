import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from pandas import Series



def plot_feature_importance(
    importances: Series,
    title: str,
    save_path: str | None = None,
    top_n: int = 10,
):
    """
    Plot horizontal bar chart of top-N feature importances.
    """
    top_features = importances.head(top_n).sort_values()

    plt.figure(figsize=(8, 6))
    top_features.plot(kind="barh")
    plt.title(title)
    plt.xlabel("Feature importance")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    plt.close()



def plot_confusion_matrix(
    cm: np.ndarray,
    class_labels: list[str] | None = None,
    title: str = "Confusion matrix",
    save_path: str | None = None,
):
    """
    Plot confusion matrix.
    """
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(cm.shape[0])
    if class_labels is None:
        class_labels = [str(i) for i in tick_marks]

    plt.xticks(tick_marks, class_labels, rotation=45)
    plt.yticks(tick_marks, class_labels)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    plt.close()
