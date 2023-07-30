from analysis.run_analyses import RunAnalyses


def main():
    RunAnalyses.run(analysis_config_name="setups/testing_config.csv")


if __name__ == "__main__":
    main()
