import os.path
import traceback

import cv2

from configs import *
import pandas as pd
from typing import cast
from multiprocessing import cpu_count, Pool
from feature_extract_helpers import extract_features_from_img
import warnings


def remove_redundant_rows(df: pd.DataFrame):
    df.sort_values(['Video Name', 'Frame Name'], ascending=[True, True], inplace=True)

    redundant_indices = []
    for idx, row in df.iterrows():
        idx = cast(int, idx)
        if idx % NB_SKIP_ROWS != 0:
            redundant_indices.append(idx)
    return df.drop(redundant_indices)


def save_metadata():
    df_list = []
    for subset, subset_meta in SUBSET_METADATA.items():
        df = pd.read_csv(subset_meta['csv_path'])
        df['Subset'] = subset
        df['Subset Path'] = subset_meta['data_dir_name']
        remove_redundant_rows(df)
        df_list.append(df)
    pd.concat(df_list).to_csv(METADATA_CSV_PATH, index=False)


def process(row):
    vid_path = DATASET_DIR_PATH
    features = []
    keys = ['Subset Path', 'Folder Name', 'Video Name']
    for key in keys:
        vid_path = os.path.join(vid_path, str(row[key]))

    frame_idx = int(row['Frame Name'].split('_')[1])

    try:
        cap = cv2.VideoCapture(vid_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            cap.release()
            raise ValueError('Cap read failed')
        cap.release()

        face_xywh = [row['X'], row['Y'], row['Width'], row['Height']]

        feature_vector = extract_features_from_img(frame, face_xywh)

        keys.extend(['Frame Name', 'X', 'Y', 'Width', 'Height', 'Liveness'])
        for key in keys:
            features.append(row[key])
        features.extend(feature_vector)

        # print(features)

    except:
        traceback.print_exc()
    return features


def extract_features_parallel():
    warnings.filterwarnings("ignore")
    metadata_df = pd.read_csv(METADATA_CSV_PATH)

    nb_frames = metadata_df.shape[0]
    shard_len = int(round(nb_frames / TOTAL_SHARDS))

    start_idx = 0

    user_frames_df_list = []
    for username in USER_ORDER:
        if username == USERNAME:
            for user_shard_idx in USER_SHARDS:
                shard_start_idx = start_idx + (user_shard_idx * shard_len)
                shard_end_idx = shard_start_idx + shard_len
                shard_end_idx = shard_end_idx if shard_end_idx < nb_frames else nb_frames
                user_frames_df_list.append(metadata_df.iloc[shard_start_idx: shard_end_idx])
            break
        else:
            start_idx += (shard_len * USER_TOTAL_SHARDS[username])

    user_df = pd.concat(user_frames_df_list)

    with Pool(cpu_count()) as p:
        data = p.map(process, user_df.to_dict('records'))
        # print(data)
        data = list(filter(lambda f: len(f) > 0, data))

        cols = ['Subset Path', 'Folder Name', 'Video Name', 'Frame Name', 'X', 'Y', 'Width', 'Height', 'Liveness']
        cols.extend([f'feature_{i + 1}' for i in range(21)])
        df = pd.DataFrame(data, columns=cols)

        OUT_CSV_FN = f'{USERNAME}_'
        for shard in USER_SHARDS:
            OUT_CSV_FN += f'{shard}_'
        OUT_CSV_FN += '.csv'
        df.to_csv(OUT_CSV_FN, index=False)


if __name__ == '__main__':
    extract_features_parallel()
