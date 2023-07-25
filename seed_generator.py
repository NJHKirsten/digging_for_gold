import json
import random


def main():
    number_of_seeds = 100
    file_path = "test_seeds.json"
    seeds = random.sample(range(0, 100000), number_of_seeds)
    json.dump(seeds, open(file_path, "w"))


if __name__ == "__main__":
    main()
