from typing import Dict, Any, AnyStr
from abc import ABC, abstractmethod
import networkx as nx

class GeneratorNoAttr(ABC):
    def __init__(
        self,
    ) -> None:
        """
        The abstract generator of graph
        """
        super().__init__()

    @abstractmethod
    def build_subgraph(self, params: Dict[AnyStr, Any]) -> nx.Graph:
        """
        Build graph of networkx.Graph type

        :param params: (Dict): Dict of required parameters for generator
        :return: (networkx.Graph): Generated graph of networkx.Graph type
        """
        raise NotImplementedError("implement run function")

