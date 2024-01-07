from typing import List, Dict

SUBSET_METADATA: Dict[str, Dict[str, str]] = {
    'train': {
        'csv_path': '../data/data.csv',
        'data_dir_name': 'train_release'
    },
    'test': {
        'csv_path': '../data/data_test.csv',
        'data_dir_name': 'test_release'
    }
}

DATASET_DIR_PATH = '../data/CASIA_faceAntisp'
METADATA_CSV_PATH = '../data/key_frames.csv'

NB_SKIP_ROWS: int = 4

# DO NOT EDIT THIS
USER_ORDER = ['hitarth', 'vaibhav', 'anik', 'jatin']
USER_TOTAL_SHARDS: Dict[str, int] = {
    'hitarth': 4,
    'vaibhav': 4,
    'anik': 14,
    'jatin': 4,
}
TOTAL_SHARDS: int = sum(USER_TOTAL_SHARDS.values())

# EDIT THIS
USERNAME = 'hitarth'
USER_SHARDS = list(range(USER_TOTAL_SHARDS[USERNAME]))




