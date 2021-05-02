from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.sampler import SubsetRandomSampler

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import os

'''
This code is adapted from two sources:
(i) The official PyTorch MNIST example (https://github.com/pytorch/examples/blob/master/mnist/main.py)
(ii) Starter code from Yisong Yue's CS 155 Course (http://www.yisongyue.com/courses/cs155/2020_winter/)
'''

class fcNet(nn.Module):
    '''
    Design your model with fully connected layers (convolutional layers are not
    allowed here). Initial model is designed to have a poor performance. These
    are the sample units you can try:
        Linear, Dropout, activation layers (ReLU, softmax)
    '''
    def __init__(self):
        # Define the units that you will use in your model
        # Note that this has nothing to do with the order in which operations
        # are applied - that is defined in the forward function below.
        super(fcNet, self).__init__()
        self.fc1 = nn.Linear(in_features=784, out_features=20)
        self.fc2 = nn.Linear(20, 10)
        self.dropout1 = nn.Dropout(p=0.5)

    def forward(self, x):
        # Define the sequence of operations your model will apply to an input x
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = F.relu(x)

        output = F.log_softmax(x, dim=1)
        return output


class ConvNet(nn.Module):
    '''
    Design your model with convolutional layers.
    '''
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), stride=1)
        self.conv2 = nn.Conv2d(8, 8, 3, 1)
        self.dropout1 = nn.Dropout2d(0.5)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(200, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        output = F.log_softmax(x, dim=1)
        return output


class Net(nn.Module):
    '''
    Build the best MNIST classifier.
    '''
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 5, 1)
        self.conv2 = nn.Conv2d(8, 8, 5, 1, 2)
        self.conv3 = nn.Conv2d(8, 16, 5, 1)
        self.drop1 = nn.Dropout2d()
        self.drop2 = nn.Dropout2d()
        self.drop3 = nn.Dropout2d()
        self.fc1 = nn.Linear(256, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 10)

    def _feature(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.drop1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.drop2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.drop3(x)
        x = F.max_pool2d(x, 2)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)

        return x

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.drop1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.drop2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.drop3(x)
        x = F.max_pool2d(x, 2)  

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)

        output = F.log_softmax(x, dim=1)
        return output



def train(args, model, device, train_loader, optimizer, epoch):
    '''
    This is your training function. When you call this function, the model is
    trained for 1 epoch.
    '''
    model.train()   # Set the model to training mode
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()               # Clear the gradient
        output = model(data)                # Make predictions
        loss = F.nll_loss(output, target)   # Compute loss
        loss.backward()                     # Gradient computation
        optimizer.step()                    # Perform a single optimization step
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.sampler),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()    # Set the model to inference mode
    test_loss = 0
    correct = 0
    test_num = 0
    with torch.no_grad():   # For the inference step, gradient is not computed
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            test_num += len(data)

    test_loss /= test_num

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, test_num,
        100. * correct / test_num))


def main():
    # Training settings
    # Use the command line to modify the default settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--step', type=int, default=1, metavar='N',
                        help='number of epochs between learning rate reductions (default: 1)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--evaluate', action='store_true', default=False,
                        help='evaluate your model on the official test set')
    parser.add_argument('--load-model', type=str,
                        help='model file path')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Evaluate on the official test set
    if args.evaluate:
        assert os.path.exists(args.load_model)

        # Set the test model
        model = Net().to(device)
        model.load_state_dict(torch.load(args.load_model))

        test_dataset = datasets.MNIST('../data', train=False,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.test_batch_size, shuffle=True, **kwargs)

        # test(model, device, test_loader)

        # To get closest images:
        # model.eval()
        # vectors = []
        # imgs = []
        # for x, y in test_loader:
        #     imgs.extend(x)
        #     vectors.extend(model._feature(x).detach().numpy())

        # vectors = np.asarray(vectors)

        # neigh = NearestNeighbors(n_neighbors=8)
        # neigh.fit(vectors)

        # neighbors = []
        # for i in range(4):
        #     vector = vectors[i]
        #     idxs = neigh.kneighbors(vector.reshape(1, -1), 9, return_distance=False)[0]
        #     temp = []
        #     temp.extend(imgs[i])
        #     for idx in idxs[1:]:
        #         v = imgs[idx].numpy()
        #         temp.extend(v)
        #     neighbors.append(temp)

        # plt.figure()
        # plt.title('Closest 8 Images to Chosen Samples')
        # for i in range(4):
        #     vector = vectors[i]
        #     for j in range(9):
        #         plt.subplot2grid((4,9), (i,j))
        #         plt.imshow(neighbors[i][j])
        # plt.show()

        # To get tSNE embeddings:
        # model.eval()
        # vectors = []
        # ys = []
        # for x, y in test_loader:
        #     vectors.extend(model._feature(x).detach().numpy())
        #     ys.extend(y.numpy())

        # embeds = TSNE(n_components=2, verbose=1).fit_transform(np.asarray(vectors))
        
        # cols = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
        #         'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
        # for c in cols:
        #     plt.plot((0, 0), color=c)
        # for em, y in zip(embeds, ys):
        #     plt.plot(em, color=cols[y])
        # plt.title('tSNE Embeddings of Digits')
        # plt.legend(range(10), bbox_to_anchor=(1, 1), loc='upper left')
        # plt.show()

        # To get confusion matrix
        # model.eval()
        # y_pred = []
        # y_true = []
        # for x, y in test_loader:
        #     pred = model(x).argmax(dim=1).numpy()
        #     y_pred.extend(pred)
        #     y_true.extend(y)
        
        # cf_matrix = confusion_matrix(y_true, y_pred)
        # print(cf_matrix)
        # plt.matshow(cf_matrix)
        # plt.xlabel('Predicted Class')
        # plt.ylabel('True Class')
        # plt.title('Confusion Matrix')
        # plt.colorbar()
        # plt.show()


        # To visuaize kernels:
        # model.eval()
        # for x, y in test_loader:
        #     for img, lbl in zip(x, y):
        #         for child in model.named_children():
        #             print(child)
        #             output = child[1](img.unsqueeze(0)).detach().numpy()[0, ...]
        #             plt.subplot(3, 3, 1)
        #             plt.imshow(img.numpy()[0, ...])
        #             plt.colorbar()
        #             for k in range(len(output)):
        #                 plt.subplot(3, 3, k + 2)
        #                 plt.imshow(output[k, ...])
        #                 plt.colorbar()
        #             plt.tight_layout()
        #             plt.show()
        #             break
        #         break
        #     break
        
        # To gather incorrect samples:
        # model.eval()
        # samples = []
        # lbls = []
        # for x, y in test_loader:
        #     for img, lbl in zip(x, y):
        #         pred = model(img.unsqueeze(0)).argmax(dim=1).numpy().squeeze()
        #         if pred != lbl.numpy().squeeze():
        #             samples.append(img.numpy()[0, ...])
        #             lbls.append((pred, lbl.numpy().squeeze()))
        #         if len(samples) == 9:
        #             break
        #     else:
        #         continue
        #     break
        
        # for i, s in enumerate(samples):
        #     plt.subplot(3, 3, i + 1)
        #     plt.imshow(s)
        #     plt.title(f'Predicted Label: {lbls[i][0]}\nActual Label: {lbls[i][1]}')
        # plt.tight_layout()
        # plt.show()

        return

    # Pytorch has default MNIST dataloader which loads data at each iteration
    train_dataset = datasets.MNIST('../data', train=True, download=True,
                transform=transforms.Compose([       # Data preprocessing
                    transforms.ToTensor(),           # Add data augmentation here
                    transforms.Normalize((0.1307,), (0.3081,))
                ]))

    # You can assign indices for training/validation or use a random subset for
    # training by using SubsetRandomSampler. Right now the train and validation
    # sets are built from the same indices - this is bad! Change it so that
    # the training and validation sets are disjoint and have the correct relative sizes.

    trans = transforms.Compose([
        transforms.Resize((31, 31), transforms.InterpolationMode.NEAREST),
        transforms.RandomCrop((28, 28)),
        transforms.RandomRotation(30),
        transforms.ToTensor()
    ])
    
    clazzes = [[] for _ in range(10)]
    for x, y in train_dataset:
        # To randomly sample subset of data:
        # if np.random.uniform() > 0.9375:
        clazzes[y].append(x.numpy())
    x_train = []
    x_val = []
    y_train = []
    y_val = []
    for i, clazz in enumerate(clazzes):
        train_split, val_split = train_test_split(clazz, test_size=0.15, random_state=42)
        # To do data augmentation:
        # train_split = list(map(lambda x: np.asarray(trans(Image.fromarray(x[0, ...]))), train_split))
        x_train.extend(train_split)
        y_train.extend(i * torch.ones(len(train_split)))
        x_val.extend(val_split)
        y_val.extend(i * torch.ones(len(val_split)))

    print('Training Samples:', len(x_train))
    x_train = torch.FloatTensor(x_train)[:len(x_train)]
    y_train = torch.LongTensor(y_train)[:len(y_train)]
    x_val = torch.FloatTensor(x_val)
    y_val = torch.LongTensor(y_val)

    train_data = torch.utils.data.TensorDataset(x_train, y_train)
    val_data = torch.utils.data.TensorDataset(x_val, y_val)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=args.batch_size, shuffle=True
    )

    # Load your model [fcNet, ConvNet, Net]
    # model = ConvNet().to(device)
    model = Net().to(device)

    # Try different optimzers here [Adam, SGD, RMSprop]
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    # Set your learning rate scheduler
    scheduler = StepLR(optimizer, step_size=args.step, gamma=args.gamma)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, val_loader)
        scheduler.step()    # learning rate scheduler

        # You may optionally save your model at each epoch here

    if args.save_model:
        torch.save(model.state_dict(), "model.pt")


if __name__ == '__main__':
    main()
