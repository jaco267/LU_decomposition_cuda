from dataclasses import dataclass
@dataclass
class TrainConfig:
  opt:int = 0
  size:int = 4
  verbose:int = 1
  pivoting:int = 0
  delete : bool = False
  check_equal: int = 0
  mc_iter: int = 1
  