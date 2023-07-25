from experiment_runner import ExperimentRunner


def main():
    ExperimentRunner.run(only_pruning=False, experiments_file="testing_config.csv")


if __name__ == "__main__":
    main()
