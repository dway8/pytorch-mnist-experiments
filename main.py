from classify_mnist import classify_mnist
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        choices=['vgg', 'simple', 'conv'], default='conv')
    parser.add_argument('--small', type=bool, default=False)
    parser.add_argument('--n_epochs', type=int,
                        help='number of epochs', default=12)
    args = parser.parse_args()

    config = dict()
    for arg in vars(args):
        config[arg] = getattr(args, arg)

    classify_mnist(config)


if __name__ == '__main__':
    main()
