import utils.utils as utils
from models.vgg_net import VGGNet
from models.conv_net import ConvNet
from models.simple_net import SimpleNet

import time
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import argparse


def classify_mnist(model_str: str):
    print(model_str)

    torch.manual_seed(42)

    PRETRAINED_SIZE = 224

    if model_str == 'vgg':
        transform = transforms.Compose([
            transforms.Resize(PRETRAINED_SIZE),
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])

    BATCH_SIZE = 32
    VALID_RATIO = 0.9

    train_data = torchvision.datasets.MNIST(
        root='./data', train=True, transform=transform, download=True)
    test_data = torchvision.datasets.MNIST(
        root='./data', train=False, transform=transform, download=True)

    n_train_examples = int(len(train_data) * VALID_RATIO)
    n_valid_examples = (len(train_data) - n_train_examples)

    train_data, valid_data = torch.utils.data.random_split(train_data,
                                                           [n_train_examples, n_valid_examples])

    # take only 10% of datasets for speed
    train_data = utils.take_percent_of_array(0.1, train_data)
    valid_data = utils.take_percent_of_array(0.1, valid_data)
    test_data = utils.take_percent_of_array(0.1, test_data)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    valid_loader = torch.utils.data.DataLoader(
        valid_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print('Train data length', len(train_data))
    print('Validation data length', len(valid_data))
    print('Test data length', len(test_data))

    classes = range(10)
    print('Classes: ', classes)

    one_feature, _ = next(iter(train_data))
    print('Shape of one feature', one_feature.shape)

    dataiter = iter(train_loader)
    images, labels = dataiter.next()

    utils.plot_images(images, labels, classes)

    loss_fn = nn.CrossEntropyLoss()

    # %load_ext tensorboard
    # %tensorboard - -logdir = runs

    if model_str == 'vgg':
        model = VGGNet()
        # freeze all layers except the last one to use the pretrained VGG model
        # TODO: freeze in the class init directly?
        for parameter in model.classifier[:-1].parameters():
            parameter.requires_grad = False
        n_epochs = 2
        log_steps = True
    elif model_str == 'simple_net':
        model = SimpleNet()
        n_epochs = 12
        log_steps = False
    elif model_str == 'conv_net':
        model = ConvNet()
        n_epochs = 12
        log_steps = False

    # default `log_dir` is "runs" - we'll be more specific here
    writer = SummaryWriter('runs/mnist_experiments')

    optimizer = optim.SGD(params=model.parameters(), lr=0.001, momentum=0.9)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(device)
    model.to(device)

    writer.add_image('mnist_random_img', images[0])

    writer.add_graph(model, images.to(device))
    writer.close()

    train_model_for_n_epochs(model, n_epochs, train_loader,
                             valid_loader, loss_fn, optimizer, device, log_steps, writer)

    test_loss, test_acc = utils.evaluate(model, test_loader, loss_fn, device)
    print(f'\t Test Loss: {test_loss:.3f} |  Test Acc: {test_acc*100:.2f}%')

    # PATH = './mnist_vgg.pth'
    # torch.save(model.state_dict(), PATH)
    # vgg_net = torchvision.models.vgg16(pretrained=True)
    # vgg_net.load_state_dict(torch.load(PATH))

    images, labels, probs = utils.get_predictions(model, test_loader, device)
    pred_labels = torch.argmax(probs, 1)
    utils.plot_confusion_matrix(labels, pred_labels, classes)

    N_IMAGES = 20
    utils.plot_most_incorrect(
        images, labels, pred_labels, probs, classes, N_IMAGES)


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
    args = parser.parse_args()

    # wrap settings into a dictionary
    config = dict()
    for arg in vars(args):
        config[arg] = getattr(args, arg)

    print(config)
    classify_mnist(config['model'])
