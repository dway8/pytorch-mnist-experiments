import random
import torch
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
        root='./downloads', train=True, transform=transform, download=True)
    test_data = torchvision.datasets.MNIST(
        root='./downloads', train=False, transform=transform, download=True)

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
