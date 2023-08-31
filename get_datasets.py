from datasets.cifar10 import Cifar10Setup

def main():
    data = Cifar10Setup().create_datasets()


if __name__ == "__main__":
    main()