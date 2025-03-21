from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class KernelMetadata:
    gb_per_s: float
    tflops: int


class Kernel(ABC):
    
    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        pass