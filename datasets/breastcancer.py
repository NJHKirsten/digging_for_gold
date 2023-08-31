import numpy as np
import pandas as pd

from dataset_setup import DatasetSetup
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class BreastCancerSetup(DatasetSetup):

    def create_datasets(self):
        data = pd.read_csv("datasets/breast_cancer_data.csv")

        features = data.iloc[:, 2:32]

        data['diagnosis'][data['diagnosis'] == 'M'] = 0
        data['diagnosis'][data['diagnosis'] == 'B'] = 1
        labels = data['diagnosis']

        train_f, test_f, train_l, test_l = train_test_split(features, labels, random_state=42)

        scaler = StandardScaler()
        train_f = scaler.fit_transform(train_f)
        test_f = scaler.transform(test_f)

        return (train_f, np.asarray(train_l).astype('float32')), (test_f, np.asarray(test_l).astype('float32'))
