import os.path
from typing import Dict, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms


def generate_indices(data_size: int, test_split: float):
    test_size = int(data_size * test_split)
    train_size = data_size - test_size
    test_indices = np.random.choice(range(data_size),
                                    size=(test_size,),
                                    replace=False)
    test_indices_mask = np.zeros(data_size, dtype=bool)
    test_indices_mask[test_indices] = True
    train_indices_mask = ~test_indices_mask

    return {
        'train': train_indices_mask,
        'test': test_indices_mask,
    }


class FaceLivenessDataset(Dataset):
    def __init__(self,
                 X: pd.DataFrame,
                 labels: pd.Series,
                 dataset_path: str,
                 transform):
        self.X = X
        self.labels = labels
        self.dataset_path = dataset_path
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        label = self.labels.iloc[idx]

        frame_idx = int(self.X.loc[self.X.index[idx], 'Frame Name'].split('_')[1])

        dir_names = ['Subset Path', 'Folder Name', 'Video Name']
        vid_fp = self.dataset_path
        for dir_name in dir_names:
            vid_fp = os.path.join(vid_fp, str(self.X.loc[self.X.index[idx], dir_name]))
        cap = cv2.VideoCapture(vid_fp)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            cap.release()
            raise ValueError('Cap read failed')
        cap.release()

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb).convert('RGB')
        pil_img_transformed = self.transform(pil_img)

        return pil_img_transformed, label


def get_datasets(dataset_path: str, csv_fp: str, usecols: List[str]):
    df = pd.read_csv(csv_fp, usecols=usecols)
    X_train = df[df['Subset Path'] == 'train_release'].copy()
    X_test = df[df['Subset Path'] == 'test_release'].copy()

    y_train = X_train['Liveness'].copy()
    y_test = X_test['Liveness'].copy()

    X_train.drop('Liveness', axis=1, inplace=True)
    X_test.drop('Liveness', axis=1, inplace=True)

    return {
        'train': FaceLivenessDataset(X_train,
                                     y_train,
                                     dataset_path,
                                     transforms.Compose(
                                         [transforms.RandomHorizontalFlip(),
                                          transforms.RandomAffine(degrees=10, translate=(.2, .2), scale=(0.75, 1.25),
                                                                  shear=[-5, 5, -5, 5]),
                                          transforms.Resize((224, 224)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])])),
        'test': FaceLivenessDataset(X_test,
                                    y_test,
                                    dataset_path,
                                    transforms.Compose([transforms.Resize((224, 224)),
                                                       transforms.ToTensor(),
                                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                             std=[0.229, 0.224, 0.225])]))
    }


if __name__ == '__main__':
    dataset_path = '../../data/CASIA_faceAntisp/'
    csv_fp = '../../data/merged_data.csv'
    usecols = ['Frame Name',
               'Video Name',
               'Folder Name',
               'Subset Path',
               'Liveness']
    data = get_datasets(dataset_path, csv_fp, usecols)
    # for i, j in data['train']:
    #     print(i.shape)
