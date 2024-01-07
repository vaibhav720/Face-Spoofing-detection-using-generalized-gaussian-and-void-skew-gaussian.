import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, SequentialSampler

from configs import BATCH_SIZE, NB_WORKERS, DEVICE
from dataset_helpers import get_datasets

MODEL_PATH = './models/model_2.pth'
dataset_path = '../../data/CASIA_faceAntisp/'
csv_fp = '../../data/merged_data.csv'
usecols = ['Frame Name',
           'Video Name',
           'Folder Name',
           'Subset Path',
           'Liveness']


def get_gt_and_probs(model, data_loader):
    model.eval()

    targets = []
    probs = []
    with torch.no_grad():
        for image, target in data_loader:
            image, target = image.to(DEVICE), target.to(DEVICE)
            prob = torch.softmax(model(image), 1)[:, 1]
            targets.append(target.cpu().numpy())
            probs.append(prob.cpu().numpy())

    targets = np.concatenate(targets)
    probs = np.concatenate(probs)
    return targets, probs


if __name__ == '__main__':
    # best_model = torch.load(MODEL_PATH, map_location='cpu')['model']
    # best_model = best_model.to(DEVICE)
    #
    # dataset = get_datasets(dataset_path, csv_fp, usecols)
    # test_ds = dataset['test']
    #
    # test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, sampler=SequentialSampler(test_ds),
    #                          num_workers=NB_WORKERS, pin_memory=True)
    #
    # y_test, y_preds_probs = get_gt_and_probs(best_model, test_loader)
    # np.save('y_test.npy', y_test)
    # np.save('y_preds_probs.npy', y_preds_probs)

    y_test = np.load('y_test.npy')
    probs = np.load('y_preds_probs.npy')

    fars = []
    frrs = []
    thresholds = np.arange(0.01, 1, .01)

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

