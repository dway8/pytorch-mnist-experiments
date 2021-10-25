
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import torch
import numpy as np
import os

default_resource_dir = 'data'
output_img_dir = os.path.join(default_resource_dir, 'output-images')


def _normalize_image(image):
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
            image = _normalize_image(image)
            image = torch.squeeze(image.permute(1, 2, 0))

        ax.imshow(image.cpu().numpy())
        ax.set_title(classes[labels[i]])
        ax.axis('off')

        os.makedirs(output_img_dir, exist_ok=True)
        plt.savefig(os.path.join(output_img_dir, 'batch-images.jpg'))


def plot_confusion_matrix(labels, pred_labels, classes):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    cm = confusion_matrix(labels, pred_labels)
    cm = ConfusionMatrixDisplay(cm, display_labels=classes)
    plt.title('Confusion Matrix')
    cm.plot(values_format='d', cmap='Greens', ax=ax)
    plt.savefig(os.path.join(output_img_dir, 'confusion-matrix.jpg'))


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
            image = _normalize_image(image)

        ax.imshow(image.cpu().numpy())
        ax.set_title(f'true label: {true_class} ({true_prob:.3f})\n'
                     f'pred label: {incorrect_class} ({incorrect_prob:.3f})')
        ax.axis('off')

    fig.subplots_adjust(hspace=0.4)
    plt.savefig(os.path.join(output_img_dir, 'most-incorrect.jpg'))
