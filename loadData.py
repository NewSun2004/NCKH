import pandas as pd

from surprise import Dataset, Reader

import os


# Tìm đường dẫn thư mục chính
base_dir = os.path.abspath(os.getcwd())


class DataLoader():
    def __init__(self, CBF_filepath: str, CF_filepath: str):
        self.meta_data = pd.read_csv(base_dir + CBF_filepath)
        self.filtered_ratings = pd.read_csv(base_dir + CF_filepath)
        reader = Reader(rating_scale=(1, 5))
        self.CF_data = Dataset.load_from_df(self.filtered_ratings, reader)
        
    def getCBFTrainSet(self):
        # Tạo trainset
        return self.CF_data.build_full_trainset()