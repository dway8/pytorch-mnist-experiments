import utils.utils as utils
import utils.plot_utils as plot_utils
from models.vgg_net import VGGNet
from models.conv_net import ConvNet
from models.simple_net import SimpleNet

import time
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import numpy as np


def classify_mnist(config: dict):
    print(f'Classification config: {config}')

    n_epochs = config['n_epochs']
    small_dataset = config['small']
    if config['model'] == 'vgg':
        model = VGGNet()
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

    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    plot_utils.plot_images(images, labels, classes)

    writer = SummaryWriter('runs/mnist_experiments')

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=model.parameters(), lr=0.001, momentum=0.9)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')
    model.to(device)

    writer.add_image('mnist_random_img', images[0])
    writer.add_graph(model, images.to(device))
    writer.close()

    _train_model(model, n_epochs, train_loader,
                 valid_loader, loss_fn, optimizer, device, log_steps, writer)

    test_loss, test_acc = _evaluate(model, test_loader, loss_fn, device)
    print(f'\t Test Loss: {test_loss:.3f} |  Test Acc: {test_acc*100:.2f}%')

    images, labels, probs = _get_predictions(model, test_loader, device)
    pred_labels = torch.argmax(probs, 1)
    plot_utils.plot_confusion_matrix(labels, pred_labels, classes)
    plot_utils.plot_most_incorrect(
        images, labels, pred_labels, probs, classes)


def _train_model(model, n_epochs, train_loader, valid_loader, loss_fn, optimizer, device, log_steps, writer=None):
    print('Starting training...')

    for epoch in range(n_epochs):
        model.train()
        start_time = time.monotonic()

        train_loss, train_accuracy = _train(
            model, train_loader, epoch, device, loss_fn, optimizer, writer, log_steps)
        valid_loss, valid_accuracy = _evaluate(
            model, valid_loader, loss_fn, device)

        end_time = time.monotonic()
        epoch_mins, epoch_secs = utils.epoch_time(start_time, end_time)

        print(
            f'Epoch: {epoch+1:01}/{n_epochs:01} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(
            f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_accuracy*100:.2f}%')
        print(
            f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_accuracy*100:.2f}%')

    print('Finished training')


def _make_train_step(model, loss_fn, optimizer):
    def train_step(x, y):
        predictions = model(x)
        loss = loss_fn(predictions, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        acc = utils.calculate_accuracy(predictions, y)
        return loss.item(), acc
    return train_step


def _make_eval_step(model, loss_fn, device):
    def eval_step(x, y):
        x = x.to(device)
        y = y.to(device)
        model.eval()
        predictions = model(x)
        loss = loss_fn(predictions, y)
        acc = utils.calculate_accuracy(predictions, y)
        return loss.item(), acc
    return eval_step


def _train(model, loader, epoch, device, loss_fn, optimizer, writer, log_steps):
    train_step = _make_train_step(model, loss_fn, optimizer)

    step_losses = []
    step_accuracies = []
    for i, data in enumerate(loader):
        x_batch, y_batch = data[0].to(device), data[1].to(device)
        step_loss, step_accuracy = train_step(x_batch, y_batch)

        if i % 50 == 49:    # print every 50 mini-batches
            if writer:
                writer.add_scalar('training loss', step_loss,
                                  epoch * len(loader) + i)
            if log_steps:
                print(
                    f'...[{epoch+1}, {i+1}] training loss: {step_loss:.3f} | training accuracy: {step_accuracy*100:.2f}%')

        step_losses.append(step_loss)
        step_accuracies.append(step_accuracy)

    loss = np.mean(step_losses)
    accuracy = np.mean(step_accuracies)

    return loss, accuracy


def _evaluate(model, loader, loss_fn, device):
    eval_step = _make_eval_step(model, loss_fn, device)

    step_losses = []
    step_accuracies = []
    with torch.no_grad():
        for x_val, y_val in loader:
            step_loss, step_accuracy = eval_step(x_val, y_val)
            step_losses.append(step_loss)
            step_accuracies.append(step_accuracy)
        loss = np.mean(step_losses)
        accuracy = np.mean(step_accuracies)
    return loss, accuracy


def _get_predictions(model, iterator, device):
    model.eval()
    images = []
    labels = []
    probs = []

    with torch.no_grad():
        for (x, y) in iterator:
            x = x.to(device)
            y_pred = model(x)

            y_prob = F.softmax(y_pred, dim=-1)

            images.append(x.cpu())
            labels.append(y.cpu())
            probs.append(y_prob.cpu())

    images = torch.cat(images, dim=0)
    labels = torch.cat(labels, dim=0)
    probs = torch.cat(probs, dim=0)

    return images, labels, probs
