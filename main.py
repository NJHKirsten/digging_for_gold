from experiment_runner import ExperimentRunner
from model_visualiser import ModelVisualiser
from run_analyses import RunAnalyses


def main():
    ExperimentRunner.run(True)
    # RunAnalyses.run()
    # ModelVisualiser.visualise()


if __name__ == "__main__":
    main()
