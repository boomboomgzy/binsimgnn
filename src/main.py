from utils import tab_printer
from param_parser import parameter_parser
from binsimgnn import BinSimGNNTrainer
import torch
import torch.multiprocessing as mp
import os


def main():
    """
    Parsing command line parameters, reading data.
    Fitting and scoring a SimGNN model.
    """
    args = parameter_parser()
    tab_printer(args)

    trainer = BinSimGNNTrainer(args)
    torch.manual_seed(18)
    torch.cuda.manual_seed_all(18)
    torch.cuda.set_device(3)
    trainer.fit()
    metric=trainer.score(mode='test')
    print(f'test metric: {str(round(metric, 10))}')

if __name__ == "__main__":
    main()
