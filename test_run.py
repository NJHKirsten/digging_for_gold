from experiment_runner import ExperimentRunner


def main():
    ExperimentRunner.run(run_name="mlp_2_hidden_test_run_lr1")
    ExperimentRunner.run(run_name="mlp_2_hidden_test_run_lr0_1")


if __name__ == "__main__":
    main()
