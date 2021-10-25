import random
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import torchvision
import torchvision.transforms as transforms

BATCH_SIZE = 32
VALID_RATIO = 0.9


def get_dataloaders(model_str: str, small_dataset: bool):

    if model_str == 'vgg':
        PRETRAINED_SIZE = 224
        transform = transforms.Compose([
            transforms.Resize(PRETRAINED_SIZE),
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                0.229, 0.224, 0.225])
        ])
    elif model_str == 'simple':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])

    elif model_str == 'conv':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])

    train_data = torchvision.datasets.MNIST(
        root='./data', train=True, transform=transform, download=True)
    test_data = torchvision.datasets.MNIST(
        root='./data', train=False, transform=transform, download=True)

    n_train_examples = int(len(train_data) * VALID_RATIO)
    n_valid_examples = (len(train_data) - n_train_examples)

    train_data, valid_data = torch.utils.data.random_split(train_data,
                                                           [n_train_examples, n_valid_examples])

    if small_dataset:
        # take only 10% of datasets for speed
        train_data = take_percent_of_array(0.1, train_data)
        valid_data = take_percent_of_array(0.1, valid_data)
        test_data = take_percent_of_array(0.1, test_data)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    valid_loader = torch.utils.data.DataLoader(
        valid_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print('Train data length', len(train_data))
    print('Validation data length', len(valid_data))
    print('Test data length', len(test_data))

    one_feature, _ = next(iter(train_data))
    print('Shape of one feature', one_feature.shape)

    return train_loader, valid_loader, test_loader


def take_percent_of_array(pc, arr):
    k = int(len(arr) * pc)
    indices = random.sample(range(len(arr)), k)
    return [arr[i] for i in indices]


def normalize_image(image):
    image_min = image.min()
    image_max = image.max()
    image.clamp_(min=image_min, max=image_max)
    image.add_(-image_min).div_(image_max - image_min + 1e-5)
    return image


def plot_images(images, labels, classes, normalize=True):

    n_images = len(images)

    rows = int(np.sqrt(n_images))
    cols = int(np.sqrt(n_images))

    fig = plt.figure(figsize=(10, 10))

    for i in range(rows*cols):

        ax = fig.add_subplot(rows, cols, i+1)

        image = images[i]

        if normalize:
            image = normalize_image(image)
            image = torch.squeeze(image.permute(1, 2, 0))

        ax.imshow(image.cpu().numpy())
        ax.set_title(classes[labels[i]])
        ax.axis('off')


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def calculate_accuracy(y_pred, y):
    _, predicted = torch.max(y_pred.data, 1)
    correct = (predicted == y).sum()
    accuracy = correct.float() / y.shape[0]
    return accuracy.cpu().numpy()


def _make_train_step(model, loss_fn, optimizer):
    def train_step(x, y):
        predictions = model(x)
        loss = loss_fn(predictions, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        acc = calculate_accuracy(predictions, y)
        return loss.item(), acc
    return train_step


def _make_eval_step(model, loss_fn, device):
    def eval_step(x, y):
        x = x.to(device)
        y = y.to(device)
        model.eval()
        predictions = model(x)
        loss = loss_fn(predictions, y)
        acc = calculate_accuracy(predictions, y)
        return loss.item(), acc
    return eval_step


def train(model, loader, epoch, device, loss_fn, optimizer, writer, log_steps):
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


def evaluate(model, loader, loss_fn, device):
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


def get_predictions(model, iterator, device):
    model.eval()

    images = []
    labels = []
    probs = []

    with torch.no_grad():

        for (x, y) in iterator:

            x = x.to(device)
            y_pred = model(x)

            y_prob = F.softmax(y_pred, dim=-1)
            top_pred = y_prob.argmax(1, keepdim=True)

            images.append(x.cpu())
            labels.append(y.cpu())
            probs.append(y_prob.cpu())

    images = torch.cat(images, dim=0)
    labels = torch.cat(labels, dim=0)
    probs = torch.cat(probs, dim=0)

    return images, labels, probs


def plot_confusion_matrix(labels, pred_labels, classes):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    cm = confusion_matrix(labels, pred_labels)
    cm = ConfusionMatrixDisplay(cm, display_labels=classes)
    cm.plot(values_format='d', cmap='Greens', ax=ax)


def _get_most_incorrect(images, labels, pred_labels, probs):
    corrects = torch.eq(labels, pred_labels)
    incorrect_examples = []

    for image, label, prob, correct in zip(images, labels, probs, corrects):
        if not correct:
            incorrect_examples.append((image, label, prob))
    incorrect_examples.sort(
        reverse=True, key=lambda x: torch.max(x[2], dim=0).values)
    return incorrect_examples


def plot_most_incorrect(images, labels, pred_labels, probs, classes, normalize=True):

    N_IMAGES = 20

    incorrect = _get_most_incorrect(images, labels, pred_labels, probs)

    rows = int(np.sqrt(N_IMAGES))
    cols = int(np.sqrt(N_IMAGES))

    fig = plt.figure(figsize=(10, 10))

    for i in range(rows*cols):

        ax = fig.add_subplot(rows, cols, i+1)

        image, true_label, probs = incorrect[i]
        image = torch.squeeze(image.permute(1, 2, 0))
        true_prob = probs[true_label]
        incorrect_prob, incorrect_label = torch.max(probs, dim=0)
        true_class = classes[true_label]
        incorrect_class = classes[incorrect_label]

        if normalize:
            image = normalize_image(image)

        ax.imshow(image.cpu().numpy())
        ax.set_title(f'true label: {true_class} ({true_prob:.3f})\n'
                     f'pred label: {incorrect_class} ({incorrect_prob:.3f})')
        ax.axis('off')

    fig.subplots_adjust(hspace=0.4)
