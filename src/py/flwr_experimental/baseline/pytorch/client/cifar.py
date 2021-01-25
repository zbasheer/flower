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
"""PyTorch CIFAR-10/100 Client for image classification."""

import argparse
import re
import timeit
from pathlib import Path
from typing import Dict, List, Type

import torch
import torch.nn as nn
import torchvision

import flwr as fl
from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, ParametersRes, Weights

from .. import DEFAULT_SERVER_ADDRESS
from ..cifar.centralized import test, train
from ..dataloader.cifar import CIFAR10PartitionedDataset, get_normalization_transform
from ..models.resnet18 import load_model

# pylint: disable=no-member
# pylint: enable=no-member


class CifarClient(fl.client.Client):
    """Flower client implementing CIFAR-10 image classification using
    PyTorch."""

    def __init__(
        self,
        *,
        cid: str,
        model: nn.Module,
        trainset: torch.utils.data.Dataset,
        testset: torch.utils.data.Dataset,
        server_address: str = DEFAULT_SERVER_ADDRESS,
        allowed_devices: List[str] = ["cpu"],
        optim_type: Type[torch.optim.Optimizer] = torch.optim.SGD,
        optim_defaults: Dict[str, float] = {"lr": 0.01},
    ):
        self.cid = cid
        self.model = model
        self.trainset = trainset
        self.testset = testset
        self.server_address = server_address
        self.allowed_devices = allowed_devices

    def _validate_device_str(self, device_str: str) -> torch.device:
        """Verifies if proposed device is allowed."""
        if device_str not in self.allowed_devices:
            device_str = "cpu"

        return torch.device(device_str)

    def get_parameters(self) -> ParametersRes:
        print(f"Client {self.cid}: get_parameters")

        weights: Weights = self.model.get_weights()
        parameters = fl.common.weights_to_parameters(weights)
        return ParametersRes(parameters=parameters)

    def fit(self, ins: FitIns) -> FitRes:
        print(f"Client {self.cid}: fit")

        weights: Weights = fl.common.parameters_to_weights(ins.parameters)
        config = ins.config
        fit_begin = timeit.default_timer()

        # Get training config
        start_epoch = int(config["global_epoch"])
        num_epochs = int(config["epochs"])
        batch_size = int(config["batch_size"])

        # Select device
        device = self._validate_device_str(str(config.get("device_str")))
        optim_dict = dict()
        optim_dict["lr"] = float(config["lr"])

        # Set model parameters
        self.model.set_weights(weights)

        # Train model
        print(
            f"Training on client {self.cid} for {num_epochs} epoch(s) from global epoch {start_epoch}."
        )
        trainloader = torch.utils.data.DataLoader(
            self.trainset, batch_size=batch_size, shuffle=True
        )
        train(
            net=self.model,
            dataloader=trainloader,
            num_epochs=num_epochs,
            device=device,
            optim_type=torch.optim.SGD,
            optim_defaults=optim_dict,
        )

        # Return the refined weights and the number of examples used for training
        weights_prime: Weights = self.model.get_weights()
        params_prime = fl.common.weights_to_parameters(weights_prime)
        num_examples_train = len(self.trainset)
        fit_duration = timeit.default_timer() - fit_begin
        return FitRes(
            parameters=params_prime,
            num_examples=num_examples_train,
            num_examples_ceil=num_examples_train,
            fit_duration=fit_duration,
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        print(f"Client {self.cid}: evaluate")

        weights = fl.common.parameters_to_weights(ins.parameters)
        # Select device
        device = self._validate_device_str(str(ins.config.get("device_str")))

        # Use provided weights to update the local model
        self.model.set_weights(weights)

        # Evaluate the updated model on the local dataset
        testloader = torch.utils.data.DataLoader(
            self.testset, batch_size=32, shuffle=False
        )
        loss, accuracy = test(net=self.model, dataloader=testloader, device=device)

        # Return the number of evaluation examples and the evaluation result (loss)
        return EvaluateRes(
            num_examples=len(self.testset), loss=float(loss), accuracy=float(accuracy)
        )


def main() -> None:
    """Load data, create and start CifarClient."""
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--server_address",
        type=str,
        default=DEFAULT_SERVER_ADDRESS,
        help=f"gRPC server address (default: {DEFAULT_SERVER_ADDRESS})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        help="Network model to be used (default: resnet18bn)",
    )
    parser.add_argument("--cid", type=str, required=True, help="Client ID (no default)")
    parser.add_argument(
        "--partitions_dir",
        type=str,
        required=True,
        help="Directory containing partitions ",
    )
    parser.add_argument(
        "--log_host",
        type=str,
        help="Logserver address (no default)",
    )
    args = parser.parse_args()

    # Configure logger
    fl.common.logger.configure(f"client_{args.cid}", host=args.log_host)

    # Load model and data
    model = load_model(num_classes=10)
    # trainset = CIFAR10PartitionedDataset(num_classes = 10,
    #    root_dir = DATA_ROOT,
    #    partition_id = int(args.client_id))

    partitions_dir = Path(args.partitions_dir).expanduser()

    train_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomCrop((24, 24)),
            torchvision.transforms.RandomHorizontalFlip(),
            get_normalization_transform(),
        ]
    )
    test_transforms = get_normalization_transform()

    trainset = CIFAR10PartitionedDataset(
        root_dir=partitions_dir / "train",
        partition_id=int(args.cid),
        transform=test_transforms,
        # transform=train_transforms,
    )
    testset = CIFAR10PartitionedDataset(
        root_dir=partitions_dir / "test",
        partition_id=int(args.cid),
        transform=test_transforms,
    )

    # Define allowed_devices
    allowed_devices: List[str] = ["cpu", "cuda"]

    # Start client
    client = CifarClient(
        cid=args.cid,
        model=model,
        trainset=trainset,
        testset=testset,
        server_address=DEFAULT_SERVER_ADDRESS,
        allowed_devices=allowed_devices,
    )

    fl.client.start_client(args.server_address, client)


if __name__ == "__main__":
    main()
