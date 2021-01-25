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
"""Minimal example on how to start a simple Flower server."""


import argparse
from typing import Callable, Dict, Optional, Tuple

import torch
import torchvision
from torchvision.datasets import CIFAR10

import flwr as fl
from flwr.common import Scalar

from .. import DEFAULT_SERVER_ADDRESS
from ..dataloader.cifar import CIFAR10PartitionedDataset, get_normalization_transform
from ..models.resnet18 import load_model
from . import DATA_ROOT
from .centralized import test, train


def fit_config(rnd: int) -> Dict[str, Scalar]:
    """Return a configuration with static batch size and (local) epochs."""
    config: Dict[str, Scalar] = {
        "global_epoch": str(rnd),
        "epochs": str(1),
        "batch_size": str(20),
        "lr": str(0.031622),
        "device_str": "cuda",
    }
    return config


def get_eval_fn(
    testset: torch.utils.data.Dataset,
) -> Callable[[fl.common.Weights], Optional[Tuple[float, float]]]:
    """Return an evaluation function for centralized evaluation."""

    def evaluate(weights: fl.common.Weights) -> Optional[Tuple[float, float]]:
        """Use the entire CIFAR-10 test set for evaluation."""
        model = load_model(num_classes=10)
        model.set_weights(weights)
        testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)
        return test(net=model, dataloader=testloader, device=torch.device("cpu"))

    return evaluate


def main() -> None:
    """Start server and train five rounds."""
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--server_address",
        type=str,
        default=DEFAULT_SERVER_ADDRESS,
        help=f"gRPC server address (default: {DEFAULT_SERVER_ADDRESS})",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="Number of rounds of federated learning (default: 1)",
    )
    parser.add_argument(
        "--sample_fraction",
        type=float,
        default=1.0,
        help="Fraction of available clients used for fit/evaluate (default: 1.0)",
    )
    parser.add_argument(
        "--min_sample_size",
        type=int,
        default=2,
        help="Minimum number of clients used for fit/evaluate (default: 2)",
    )
    parser.add_argument(
        "--min_num_clients",
        type=int,
        default=2,
        help="Minimum number of available clients required for sampling (default: 2)",
    )
    parser.add_argument(
        "--log_host",
        type=str,
        help="Logserver address (no default)",
    )
    args = parser.parse_args()

    # Load evaluation data
    # _, testset = cifar.load_data()
    testset = CIFAR10(
        root=DATA_ROOT,
        download=True,
        train=False,
        transform=get_normalization_transform(),
    )

    # Create strategy
    """strategy = fl.server.strategy.FedAvg(
        fraction_fit=args.sample_fraction,
        min_fit_clients=args.min_sample_size,
        min_available_clients=args.min_num_clients,
        eval_fn=get_eval_fn(testset),
        on_fit_config_fn=fit_config,
    )"""

    # Create model
    temp_model = load_model(num_classes=10)
    initial_weights: fl.common.Weights = temp_model.get_weights()

    strategy = fl.server.strategy.FedAdagrad(
        current_weights=initial_weights,
        fraction_fit=1,
        fraction_eval=1,
        min_fit_clients=2,
        min_eval_clients=2,
        min_available_clients=2,
        eval_fn=get_eval_fn(testset),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=None,
        accept_failures=True,
        eta=0.1,
        eta_l=0.031622,
        tau=0.01,
    )

    # Configure logger and start server
    fl.common.logger.configure("server", host=args.log_host)
    fl.server.start_server(
        args.server_address,
        config={"num_rounds": args.rounds},
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
