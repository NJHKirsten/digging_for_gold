import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

from dataset_setup import DatasetSetup
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class BreastCancerSetup(DatasetSetup):

    def create_datasets(self):
        data = pd.read_csv("datasets/breast_cancer_data.csv")

        features = data.iloc[:, 2:32]
        labels = data['diagnosis']
        labels[labels == 'M'] = 0
        labels[labels == 'B'] = 1
        labels = data['diagnosis']

        train_f, test_f, train_l, test_l = train_test_split(features, labels, random_state=42)

        scaler = StandardScaler()
        train_f = scaler.fit_transform(train_f)
        test_f = scaler.transform(test_f)

        train_dataset = TensorDataset(torch.from_numpy(train_f.astype('float32')),
                                      torch.from_numpy(np.asarray(train_l).reshape([-1, 1]).astype('float32')))
        test_dataset = TensorDataset(torch.from_numpy(test_f.astype('float32')),
                                     torch.from_numpy(np.asarray(test_l).reshape([-1, 1]).astype('float32')))

        return train_dataset, test_dataset
