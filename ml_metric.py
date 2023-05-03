from dataclasses import dataclass
from typing import Optional


@dataclass
class MlMetric:
    accuracy: float
    precision: float
    recall: float
    specificity: float
    loss: Optional[float] = None
    F1: Optional[float] = None
    predictions: Optional[list] = None
    labels: Optional[list] = None
