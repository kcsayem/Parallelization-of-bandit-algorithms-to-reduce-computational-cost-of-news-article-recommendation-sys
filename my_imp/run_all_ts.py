from parallel_lazy_tscontext_yahoo import experiment as lazy_exp
from parallel_nonlazy_tscontext_yahoo import experiment as nonlazy_exp
import numpy as np
import warnings
from datetime import datetime
if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    SEED = 42
    np.random.seed(SEED)
    path = "data/R6A_spec"
    print("================================================")
    print("RUNNING LAZY EXPERIMENTS")
    print("================================================")
    for p in [25, 50, 100]:
        print("================================================")
        print(f"TRYING WITH P = {p}")
        start = datetime.now()
        lazy_exp(path,p)
        end = datetime.now()
        print(f"DURATION FOR P = {p}: {end - start}")
        print("================================================")
    print()
    print()
    print("================================================")
    print("RUNNING NON-LAZY EXPERIMENTS")
    print("================================================")
    for p in [25, 50, 100]:
        print("================================================")
        print(f"TRYING WITH P = {p}")
        start = datetime.now()
        nonlazy_exp(path,p)
        end = datetime.now()
        print(f"DURATION FOR P = {p}: {end - start}")
        print("================================================")
