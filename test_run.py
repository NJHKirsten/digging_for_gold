from experiment_runner import ExperimentRunner


def main():
    ExperimentRunner.run(run_name="mlp_4_hidden_test_run_lr0_01")
    ExperimentRunner.run(run_name="mlp_4_hidden_test_run_lr0_001")


if __name__ == "__main__":
    main()
