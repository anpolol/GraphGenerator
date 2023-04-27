import numpy as np
from typing import Tuple

def sum(min_d: int, max_d: int) -> float:
    """
    Calculate the sum of inverse power for power degree distribution

    :param min_d: (int): Degree value of the node with the minimum degree
    :param max_d: (int): Degree value of the node with the maximum degree
    :return: (float): The sum
    """
    sum = float(np.sum([1 / (pow(i, 2.0)) for i in range(min_d, max_d + 1)]))
    return sum


def pk(min_d: int, max_d: int, power: float) -> Tuple[float]:
    """
    Build a power distribution of degrees

    :param min_d: (int): Degree value of the node with the minimum degree
    :param max_d: (int): Degree value of the node with the maximum degree
    :return: (Tuple[float]): The power degree distribution
    """
    probs = []
    sum = sum(min_d, max_d)
    for x in range(min_d, max_d + 1):
        probability = 1 / (pow(x, power) * sum)
        probs.append(probability)
    return tuple(probs)