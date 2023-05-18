from typing import Tuple

import numpy as np
import torch


def sum(min_d: int, max_d: int, power: float) -> float:
    """
    Calculate the sum of inverse power for power degree distribution

    :param min_d: (int): Degree value of the node with the minimum degree
    :param max_d: (int): Degree value of the node with the maximum degree
    :return: (float): The sum
    """
    sum = float(np.sum([1 / (pow(i, power)) for i in range(min_d, max_d + 1)]))
    return sum


def pk(min_d: int, max_d: int, power: float) -> Tuple[float]:
    """
    Build a power distribution of degrees

    :param min_d: (int): Degree value of the node with the minimum degree
    :param max_d: (int): Degree value of the node with the maximum degree
    :return: (Tuple[float]): The power degree distribution
    """
    probs = []
    summation = sum(min_d, max_d, power)
    for x in range(min_d, max_d + 1):
        probability = 1 / (pow(x, power) * summation)
        probs.append(probability)
    return tuple(probs)


def cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Calculate cos between two vectors a and b

    :param a: (torch.Tensor): First tensor
    :param b: (torch.Tensor): Second tensor
    :return: torch.Tensor: One value of cos between a and b
    """
    return (torch.matmul(a, b)) / (torch.norm(a) * torch.norm(b))
