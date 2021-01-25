import argparse
from os import PathLike
from pathlib import Path

from numpy import ndarray
from torch import load, save
from torchvision.datasets import CIFAR10, CIFAR100

from flwr.dataset.utils.common import create_lda_partitions
from flwr.dataset.utils.pytorch import convert_torchvision_dataset_to_xy

DATA_ROOT: str = "~/.flower/data/cifar"

if __name__ == "__main__":
    """Generates Latent Dirichlet Allocated Partitions for CIFAR10/100
    datasets.

    This script does NOT perform any kind of data normalization. As
    described in <https://www.cs.toronto.edu/~kriz/cifar.html>, outputs
    are in the for (C,H,W)
    """
    parser = argparse.ArgumentParser(
        description="Generate Latent Dirichlet Allocated Partitions for CIFAR10/100 datasets."
    )

    parser.add_argument(
        "--num_classes",
        type=int,
        required=True,
        choices=[10, 100],
        help="Choose 10 for CIFAR10 and 100 for CIFAR100.",
    )
    parser.add_argument(
        "--num_partitions",
        type=int,
        default=500,
        help="Number of partitions in which to split the dataset.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Choose Dirichlet concentration.",
    )
    parser.add_argument(
        "--save_root",
        type=str,
        default=DATA_ROOT,
        help="Choose where to save partition.",
    )

    args = parser.parse_args()
    save_original_root: Path = (
        Path(f"{args.save_root}").expanduser() / f"cifar{args.num_classes}"
    )
    save_root: Path = save_original_root / "partitions" / "lda" / f"{args.alpha:.2f}"
    if save_root.is_dir():
        raise OSError(
            f"Directory {save_root} is not empty. Please empty it before generating partition."
        )

    if args.num_classes == 10:
        CIFAR = CIFAR10
    else:
        CIFAR == CIFAR100

    train_dataset = CIFAR(
        root=args.save_root,
        train=True,
        download=True,
    )
    test_dataset = CIFAR(
        root=args.save_root,
        train=False,
        download=True,
    )

    # Generate distributions
    dist: ndarray = None
    for dataset, data_str in [(train_dataset, "train"), (test_dataset, "test")]:
        save_dir = save_root / data_str
        save_dir.mkdir(parents=True, exist_ok=True)

        np_dataset = convert_torchvision_dataset_to_xy(dataset)

        partitions, dist = create_lda_partitions(
            dataset=np_dataset,
            dirichlet_dist=dist,
            num_partitions=args.num_partitions,
            concentration=args.alpha,
        )

        for idx, part in enumerate(partitions):
            save(part, save_dir / f"{idx}.pt")
