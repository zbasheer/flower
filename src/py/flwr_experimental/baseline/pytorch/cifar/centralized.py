# Copyright 2020 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Centralized training for CIFAR-10/100."""
from typing import Dict, Tuple, Type

import torch
import torchvision
from torch import LongTensor, Tensor, device, nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18
from tqdm import tqdm

from ..dataloader.cifar import CIFAR_PartitionedDataset, get_normalization_transform
from ..models.resnet18 import load_model
from . import DATA_ROOT


def train(
    *,
    net: nn.Module,
    dataloader: "DataLoader[Tuple[Tensor, LongTensor]]",
    num_epochs: int,
    device: torch.device,  # pylint: disable=no-member
    optim_type: Type[torch.optim.Optimizer] = torch.optim.SGD,
    optim_defaults: Dict[str, float] = {"lr": 0.01},
) -> None:
    """Train the network."""
    net.train()
    # Define loss and optimizer
    criterion = nn.modules.loss.CrossEntropyLoss()
    net = net.to(device)
    optimizer = optim_type(net.parameters(), **optim_defaults)

    # Train the network
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        total, running_loss = 0, 0.0
        for idx, data in enumerate(dataloader):
            images, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            total +=labels.size(0)
        running_loss /= total
        print(f"Train|Loss:{loss}|")
        running_loss = 0.0


def test(
    *,
    net: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,  # pylint: disable=no-member
) -> Tuple[float, float]:
    """Validate the network on the entire test set."""
    net.eval()
    net = net.to(device)
    criterion = nn.modules.loss.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    print("Evaluating...")
    with torch.no_grad():
        for data in dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(
                input=outputs.data, dim=1
            )  # pylint: disable-msg=no-member
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f"Test|Loss:{loss}|Accuracy:{accuracy}|")
    return loss, accuracy

train_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomCrop((24, 24)),
            torchvision.transforms.RandomHorizontalFlip(),
            get_normalization_transform(),
        ]
    )
test_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomCrop((24, 24)),
            get_normalization_transform(),
        ]
    )

if __name__ == "__main__":

    net = load_model(num_classes=10)
    print(net)

    trainset = CIFAR10(
        root=DATA_ROOT,
        download=True,
        train=True,
        transform=train_transforms,
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=20, shuffle=True, num_workers=2
    )

    testset = CIFAR10(
        root=DATA_ROOT,
        download=True,
        train=False,
        transform=test_transforms,
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=32, shuffle=True, num_workers=2
    )

    device = torch.device("cuda:0")
    loss, accuracy = test(net=net, dataloader=testloader, device=device)
    print(f"Initial Loss: {loss:.2f} | Initial accuracy: {accuracy:.2f}")
    for epoch, _ in enumerate(range(30)):
        train(net=net, dataloader=trainloader, num_epochs=1, device=device)
        loss, accuracy = test(net=net, dataloader=testloader, device=device)
