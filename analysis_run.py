import warnings

from analysis.run_analyses import RunAnalyses


def main():
    # warnings.simplefilter('error', UserWarning)
    RunAnalyses.run("mlp_2_hidden_analysis")


if __name__ == "__main__":
    main()
