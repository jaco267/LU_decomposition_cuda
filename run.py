import os
import pyrallis
from config import TrainConfig
def c(cmd_str):
    os.system(cmd_str)

@pyrallis.wrap()    
def main(cfg: TrainConfig):
  if cfg.delete:
    c("rm -r ./build")
  c("mkdir ./build")
  os.chdir("./build")
  c(f"""cmake ..""")
  c("cmake  --build . --config Release")
  # run_cmd = f"./main -o {cfg.opt}"
  run_cmd = f"./main  -o {cfg.opt} -s {cfg.size} -v \
    {cfg.verbose} -p {cfg.pivoting} -e {cfg.check_equal} -m {cfg.mc_iter}"
  c(run_cmd)
  
if __name__ == '__main__':
   main()