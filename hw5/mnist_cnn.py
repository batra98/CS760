'''
BSD 3-Clause License

Copyright (c) 2017, Pytorch contributors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        ####################################################################
        # TODO: Define the neural network architecture with these specifications:
        # 1. First convolutional layer (conv1):
        #    - input channels: 1 (grayscale image)
        #    - output channels: 32
        #    - kernel size: 3
        #    - stride: 1
        self.conv1 = # Define the first convolutional layer here

        # 2. Second convolutional layer (conv2):
        #    - input channels: 32 (from conv1)
        #    - output channels: 64
        #    - kernel size: 3
        #    - stride: 1
        self.conv2 = # Define the second convolutional layer here

        # 3. First dropout layer (dropout1):
        #    - dropout rate: 0.25
        self.dropout1 = # Define the first dropout layer here

        # 4. Second dropout layer (dropout2):
        #    - dropout rate: 0.5
        self.dropout2 = # Define the second dropout layer here

        # 5. First fully connected layer (fc1):
        #    - input features: 9216 (64 * 12 * 12)
        #    - output features: 128
        self.fc1 = # Define the first fully connected layer here

        # 6. Second fully connected layer (fc2):
        #    - input features: 128
        #    - output features: 10 (number of classes)
        self.fc2 = # Define the second fully connected layer here
        ####################################################################
        pass

    def forward(self, x):
        ####################################################################
        # TODO: Implement the forward pass with these exact steps:
        # 1. Apply conv1 and ReLU activation
        # 2. Apply conv2 and ReLU activation
        # 3. Apply max pooling with kernel size 2
        # 4. Apply dropout1
        # 5. Flatten the tensor (hint: use torch.flatten(x, 1))
        # 6. Apply fc1 and ReLU activation
        # 7. Apply dropout2
        # 8. Apply fc2
        # 9. Apply log_softmax with dim=1
        ####################################################################
        pass


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        ####################################################################
        # TODO: Implement the training step:
        # 1. Zero the gradients with optimizer.zero_grad()
        # 2. Perform forward pass through the model
        # 3. Calculate the loss using F.nll_loss(output, target)
        # 4. Perform backward pass with loss.backward()
        # 5. Update weights with optimizer.step()
        ####################################################################

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            ####################################################################
            # TODO: Implement the testing step:
            # 1. Perform forward pass through the model
            # 2. Calculate and accumulate loss using F.nll_loss(output, target, reduction='sum').item()
            # 3. Get predictions using output.argmax(dim=1, keepdim=True)
            # 4. Calculate number of correct predictions using pred.eq(target.view_as(pred)).sum().item()
            ####################################################################

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args, _ = parser.parse_known_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                        'pin_memory': True,
                        'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                            transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                            transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    #######################################################################
    # TODO: Initialize the Adadelta optimizer (https://pytorch.org/docs/stable/generated/torch.optim.Adadelta.html)
    # Parameters:
    # - model.parameters()
    # - learning rate: args.lr
    optimizer = # Initialize the Adadelta optimizer here
    #######################################################################

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
