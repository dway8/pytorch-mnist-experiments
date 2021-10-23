import utils.utils as utils

import time
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


torch.manual_seed(42)

PRETRAINED_SIZE = 224

transform = transforms.Compose([
    transforms.Resize(PRETRAINED_SIZE),
    transforms.Grayscale(3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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

print(len(train_data))
print(len(valid_data))
print(len(test_data))

classes = range(10)
print(classes)

dataiter = iter(train_loader)
images, labels = dataiter.next()

utils.plot_images(images, labels, classes)

# 1. Pretrained VGG16

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(params=vgg_net.parameters(), lr=0.001, momentum=0.9)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)
vgg_net.to(device)

print("origin", vgg_net.classifier[-1])
IN_FEATURES = vgg_net.classifier[-1].in_features
final_layer = nn.Linear(IN_FEATURES, 10)
vgg_net.classifier[-1] = final_layer
print("final", vgg_net.classifier[-1])

# torch.cuda.empty_cache()
torch.cuda.memory_summary(device=None, abbreviated=False)

for parameter in vgg_net.classifier[:-1].parameters():
    parameter.requires_grad = False


def train_model_for_n_epochs(model, n_epochs, train_loader, valid_loader, loss_fn, optimizer, device, writer=None, log_steps=True):

    for epoch in range(n_epochs):
        model.train()

        start_time = time.monotonic()

        train_loss, train_accuracy = train(
            model, train_loader, epoch, device, loss_fn, optimizer, writer, log_steps)
        valid_loss, valid_accuracy = evaluate(model, valid_loader, loss_fn)

        end_time = time.monotonic()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f'Epoch: {epoch+1:01} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(
            f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_accuracy*100:.2f}%')
        print(
            f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_accuracy*100:.2f}%')

    print('Finished Training')


N_EPOCHS = 2
train_model_for_n_epochs(vgg_net, N_EPOCHS, train_loader,
                         valid_loader, loss_fn, optimizer, device)

test_loss, test_acc = evaluate(vgg_net, test_loader, loss_fn)
print(f'\t Test Loss: {test_loss:.3f} |  Test Acc: {test_acc*100:.2f}%')

PATH = './mnist_vgg.pth'
torch.save(vgg_net.state_dict(), PATH)

#vgg_net = torchvision.models.vgg16(pretrained=True)
# vgg_net.load_state_dict(torch.load(PATH))

images, labels, probs = get_predictions(vgg_net, test_loader)
pred_labels = torch.argmax(probs, 1)
plot_confusion_matrix(labels, pred_labels, classes)

N_IMAGES = 20
plot_most_incorrect(images, labels, probs, classes, N_IMAGES)

# 2. Feed forward net

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

print(len(train_data))
print(len(valid_data))
print(len(test_data))

one_feature, _ = next(iter(train_data))
one_feature.shape

dataiter = iter(train_loader)
images, labels = dataiter.next()

plot_images(images, labels, classes)


net = Net()
net.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(params=net.parameters(), lr=0.001, momentum=0.9)

N_EPOCHS = 12
train_model_for_n_epochs(net, N_EPOCHS, train_loader,
                         valid_loader, loss_fn, optimizer, device, log_steps=False)

test_loss, test_acc = evaluate(net, test_loader, loss_fn)
print(f'\t Test Loss: {test_loss:.3f} |  Test Acc: {test_acc*100:.2f}%')

PATH = './mnist_net.pth'
torch.save(net.state_dict(), PATH)

images, labels, probs = get_predictions(net, test_loader)
pred_labels = torch.argmax(probs, 1)
plot_confusion_matrix(labels, pred_labels, classes)

N_IMAGES = 20
plot_most_incorrect(images, labels, probs, classes, N_IMAGES)

# 3. CNN

conv_net = ConvNet()
conv_net.to(device)

%load_ext tensorboard
%tensorboard - -logdir = runs

!kill 704

dataiter = iter(train_loader)
images, labels = dataiter.next()

#plot_images(images, labels, classes)
writer.add_image('mnist_random_img', images[0])

writer.add_graph(conv_net, images.to(device))
writer.close()

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(params=conv_net.parameters(), lr=0.001, momentum=0.9)

N_EPOCHS = 12
train_model_for_n_epochs(conv_net, N_EPOCHS, train_loader, valid_loader,
                         loss_fn, optimizer, device, writer, log_steps=False)

test_loss, test_acc = evaluate(conv_net, test_loader, loss_fn)
print(f'\t Test Loss: {test_loss:.3f} |  Test Acc: {test_acc*100:.2f}%')

PATH = './mnist_conv_net.pth'
torch.save(conv_net.state_dict(), PATH)

images, labels, probs = get_predictions(conv_net, test_loader)
pred_labels = torch.argmax(probs, 1)
plot_confusion_matrix(labels, pred_labels, classes)

N_IMAGES = 20
plot_most_incorrect(images, labels, probs, classes, N_IMAGES)
