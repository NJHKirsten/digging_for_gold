from abc import ABC, abstractmethod


class Analysis(ABC):

    def __init__(self, analysis_config, run_config, seeds_file):
        self.analysis_config = analysis_config
        self.run_config = run_config
        self.seeds_file = seeds_file

    @abstractmethod
    def run(self):
        pass
