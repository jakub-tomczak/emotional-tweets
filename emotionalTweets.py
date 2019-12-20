from data_loader import load_dataset


def main():
    train, test = load_dataset()
    print(train.shape, test.shape)


if __name__ == "__main__":
    main()
