"""Getting params from the command line."""

import argparse

def parameter_parser():
    """
    A method to parse up command line parameters.
    The default hyperparameters give a high performance model without grid search.
    """
    parser = argparse.ArgumentParser(description="Run SimGNN.")
    parser.add_argument("--debug-training-graphs",
                        nargs="?",
                        default="/home/ouyangchao/binsimgnn/binkit_small_heteroG_dataset/train/",
	                help="Folder with training graph pair jsons.")

    parser.add_argument("--debug-testing-graphs",
                        nargs="?",
                        default="/home/ouyangchao/binsimgnn/binkit_small_heteroG_dataset/test/",
	                help="Folder with testing graph pair jsons.")
    
    parser.add_argument("--training-graphs",
                        nargs="?",
                        default="./heteroG_dataset/train/",
	                help="Folder with training graph pair jsons.")

    parser.add_argument("--testing-graphs",
                        nargs="?",
                        default="./heteroG_dataset/test/",
	                help="Folder with testing graph pair jsons.")

    parser.add_argument("--epochs",
                        type=int,
                        default=100,
	                help="Number of training epochs. Default is 100.")

    parser.add_argument("--tensor-neurons",
                        type=int,
                        default=16,
	                help="Neurons in tensor network layer. Default is 16.") #NTN中W切片的数量

    parser.add_argument("--bottle-neck-neurons",
                        type=int,
                        default=16,
	                help="Bottle neck layer neurons. Default is 16.")

    parser.add_argument("--num_layers",
                    type=int,
                    default=2,
                help="GCN layers num.")


    parser.add_argument("--hidden_dim",
                    type=int,
                    default=64,
                help="hidden layer dimension")

    parser.add_argument("--heads",
                    type=int,
                    default=3,
                help="attention heads")
    
    parser.add_argument("--batch-size",  
                        type=int,
                        default=4,
	                help="Number of graph pairs per batch, must be even . Default is 4.")

    parser.add_argument("--bins",
                        type=int,
                        default=16,
	                help="Similarity score bins. Default is 16.")

    parser.add_argument("--dropout",
                        type=float,
                        default=0.5,
	                help="Dropout probability. Default is 0.5.")

    parser.add_argument("--learning-rate",
                        type=float,
                        default=0.005,
	                help="Learning rate. Default is 0.001.")

    parser.add_argument("--weight-decay",
                        type=float,
                        default=5*10**-4,
	                help="Adam weight decay. Default is 5*10^-4.")

    parser.add_argument("--save-dir",
                        type=str,
                        default=r'/home/ouyangchao/binsimgnn/model',
                        help="Dir to save the trained model")

    parser.add_argument("--load-path",
                        type=str,
                        default=None,
                        help="Load a pretrained model")

    return parser.parse_args()
