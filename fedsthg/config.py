"""Paper settings and explicitly configurable B/C assumptions."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass
class Config:
    dataset: str = "ABIDE"
    n_rois: int = 116
    aligned_length: int = 116
    window: int = 30                 # B01: not specified in paper
    step: int = 2                    # B01
    hidden: int = 64
    layers: int = 2
    k_sg: int = 8
    k_tg: int = 5
    k_sh: int = 10
    k_th: int = 7
    k_subject: int = 5               # B04
    dropout: float = 0.3
    lr: float = 3e-4
    weight_decay: float = 1e-4
    clip: float = 1.0
    consistency: float = 1e-3
    proximal: float = 1.0
    batch_size: int = 16
    rounds: int = 50
    seed: int = 42                   # B08
    folds: int = 5
    validation_fraction: float = .2
    variant: str = "full"

    @property
    def windows(self):
        return (self.aligned_length - self.window) // self.step + 1

    def validate(self):
        if not (2 <= self.window <= self.aligned_length and self.step > 0):
            raise ValueError("Require 2 <= window <= aligned_length and step > 0")
        if self.n_rois < 2 or self.hidden < 2 or self.layers < 1:
            raise ValueError("Invalid network dimensions")
        for k, n in [(self.k_sg, self.n_rois), (self.k_sh, self.n_rois),
                     (self.k_tg, self.windows), (self.k_th, self.windows)]:
            if not 0 <= k < n:
                raise ValueError(f"Neighborhood {k} must be smaller than node count {n}")
        if self.k_subject < 0 or not 0 <= self.dropout < 1:
            raise ValueError("Invalid subject k or dropout")
        if self.batch_size < 1 or self.rounds < 1 or self.folds < 2:
            raise ValueError("Invalid training settings")
        if self.consistency < 0 or self.proximal < 0:
            raise ValueError("Regularization must be nonnegative")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("Invalid validation fraction")
        if self.variant not in {"full", "no_temporal", "no_spatial", "graph_only",
                                "hypergraph_only", "equal_fusion", "no_personalization"}:
            raise ValueError("Unknown ablation")
        return self

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8"))).validate()
