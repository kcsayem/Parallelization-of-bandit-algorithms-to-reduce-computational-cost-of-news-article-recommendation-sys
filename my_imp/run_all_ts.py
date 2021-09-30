from parallel_lazy_tscontext_yahoo import experiment as lazy_exp
from parallel_nonlazy_tscontext_yahoo import experiment as nonlazy_exp
from helper_functions import print_example_banner as ptb
import numpy as np
import warnings
from datetime import datetime

def experiment(processes):
    warnings.filterwarnings('ignore')
    SEED = 42
    np.random.seed(SEED)
    path = "data/R6A_spec"
    ptb("RUNNING LAZY EXPERIMENTS")
    for p in processes:
        ptb(f"TRYING WITH P = {p}")
        start = datetime.now()
        lazy_exp(path, p)
        end = datetime.now()
        ptb(f"DURATION FOR P = {p}: {end - start}")
    print()
    print()
    ptb("RUNNING NON-LAZY EXPERIMENTS")
    for p in processes:
        ptb(f"TRYING WITH P = {p}")
        start = datetime.now()
        nonlazy_exp(path, p)
        end = datetime.now()
        ptb(f"DURATION FOR P = {p}: {end - start}")

if __name__ == "__main__":
    experiment([500,1000,2000])