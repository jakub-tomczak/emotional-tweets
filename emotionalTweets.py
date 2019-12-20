from data_loader import load_dataset
from nltk_helper import initialize_tokenizer


def main():
    train, test = load_dataset()

    if not initialize_tokenizer():
        exit(1)


if __name__ == "__main__":
    main()
