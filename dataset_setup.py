from abc import abstractmethod


class DatasetSetup:

    @abstractmethod
    def create_datasets(self):
        pass
