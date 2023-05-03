from dataclasses import dataclass
from typing import Optional

@dataclass
class MlMetric():
    accuracy: float 
    precision: float
    recall: float
    specificity: float
    predictions: Optional[list] = None
    labels: Optional[list] = None
