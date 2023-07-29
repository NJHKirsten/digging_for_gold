from abc import ABC, abstractmethod


class Analysis(ABC):

    def __init__(self, model_name, model_class, dataset_name, dataset_class, sample_size, run_config):
        self.model_name = model_name
        self.model_class = model_class
        self.dataset_name = dataset_name
        self.dataset_class = dataset_class
        self.sample_size = sample_size
        self.run_config = run_config

    @abstractmethod
    def run(self):
        pass
