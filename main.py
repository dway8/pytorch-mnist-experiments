import utils.utils as utils
from models.vgg_net import VGGNet
from models.conv_net import ConvNet
from models.simple_net import SimpleNet

import time
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import argparse


def classify_mnist(config: dict):
    print('Classification config', config)

    n_epochs = config['n_epochs']
    small_dataset = config['small']
    if config['model'] == 'vgg':
        model = VGGNet()
        n_epochs = 2
        log_steps = True
    elif config['model'] == 'simple':
        model = SimpleNet()
        log_steps = False

    elif config['model'] == 'conv':
        model = ConvNet()
        log_steps = False
    else:
        print(f'Unknown model {config["model"]}, exiting')
        return

    torch.manual_seed(42)

    train_loader, valid_loader, test_loader = utils.get_dataloaders(config['model'],
                                                                    small_dataset)

    classes = range(10)
    print('Classes: ', classes)

    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    utils.plot_images(images, labels, classes)

    writer = SummaryWriter('runs/mnist_experiments')

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=model.parameters(), lr=0.001, momentum=0.9)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')
    model.to(device)

    writer.add_image('mnist_random_img', images[0])
    writer.add_graph(model, images.to(device))
    writer.close()

    train_model_for_n_epochs(model, n_epochs, train_loader,
                             valid_loader, loss_fn, optimizer, device, log_steps, writer)

    test_loss, test_acc = utils.evaluate(model, test_loader, loss_fn, device)
    print(f'\t Test Loss: {test_loss:.3f} |  Test Acc: {test_acc*100:.2f}%')

    images, labels, probs = utils.get_predictions(model, test_loader, device)
    pred_labels = torch.argmax(probs, 1)
    utils.plot_confusion_matrix(labels, pred_labels, classes)

    utils.plot_most_incorrect(
        images, labels, pred_labels, probs, classes)


def train_model_for_n_epochs(model, n_epochs, train_loader, valid_loader, loss_fn, optimizer, device, log_steps, writer=None):

    for epoch in range(n_epochs):
        model.train()

        start_time = time.monotonic()

        train_loss, train_accuracy = utils.train(
            model, train_loader, epoch, device, loss_fn, optimizer, writer, log_steps)
        valid_loss, valid_accuracy = utils.evaluate(
            model, valid_loader, loss_fn, device)

        end_time = time.monotonic()

        epoch_mins, epoch_secs = utils.epoch_time(start_time, end_time)

        print(
            f'Epoch: {epoch+1:01} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(
            f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_accuracy*100:.2f}%')
        print(
            f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_accuracy*100:.2f}%')

    print('Finished Training')


if __name__ == '__main__':

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
