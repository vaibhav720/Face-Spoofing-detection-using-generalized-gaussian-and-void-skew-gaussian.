import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from joblib import parallel_backend


def train_model(model, csv_fp, redundant_cols):
    df = pd.read_csv(csv_fp)

    df.drop(columns=redundant_cols, errors='ignore', inplace=True)

    X_train = df.loc[df['Subset Path'] == 'train_release'].copy()
    X_test = df.loc[df['Subset Path'] == 'test_release'].copy()

    y_train = X_train['Liveness'].copy()
    X_train.drop(['Liveness', 'Subset Path'], axis=1, inplace=True)

    y_test = X_test['Liveness'].copy()
    X_test.drop(['Liveness', 'Subset Path'], axis=1, inplace=True)

    # X = df.drop('Liveness', axis=1)
    # y = df['Liveness']

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)

    with parallel_backend('threading', n_jobs=-1):
        model.fit(X_train, y_train)

        fars = []
        frrs = []
        thresholds = np.arange(0, 1, .01)
        probs = model.predict_proba(X_test)[:, 1]

        min_difference = 500
        eer = -1
        eer_threshold = -1
        for threshold in thresholds:
            y_pred = np.zeros_like(probs, dtype=np.int32)
            y_pred[probs > threshold] = 1
            confusion_mat = confusion_matrix(y_test, y_pred)
            far = confusion_mat[0, 1] / (confusion_mat[0, 1] + confusion_mat[0, 0])
            frr = confusion_mat[1, 0] / (confusion_mat[1, 0] + confusion_mat[1, 1])
            fars.append(far)
            frrs.append(frr)
            if abs(far - frr) < min_difference:
                min_difference = abs(far - frr)
                eer = (far + frr) / 2
                eer_threshold = threshold
        plt.plot(thresholds, fars, label="FAR")
        plt.plot(thresholds, frrs, label="FRR")
        plt.legend(loc="upper left")
        plt.show()

        print(eer_threshold, min_difference)
        print(f"EER: {eer * 100:.2f}%")

        y_pred = np.zeros_like(probs, dtype=np.int32)
        y_pred[probs > .5] = 1
        confusion_mat = confusion_matrix(y_test, y_pred)
        far = confusion_mat[0, 1] / (confusion_mat[0, 1] + confusion_mat[0, 0])
        frr = confusion_mat[1, 0] / (confusion_mat[1, 0] + confusion_mat[1, 1])
        print(f"HTER: {((far + frr) / 2) * 100:.2f}%")



if __name__ == '__main__':
    train_model(RandomForestClassifier(n_estimators=100, random_state=42),
                '../../data/merged_data.csv',
                ['Height',
                 'Width',
                 'X',
                 'Y',
                 'Frame Name',
                 'Video Name',
                 'Folder Name'])
