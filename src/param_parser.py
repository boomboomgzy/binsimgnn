"""Getting params from the command line."""

import argparse

def parameter_parser():
    """
    A method to parse up command line parameters.
    The default hyperparameters give a high performance model without grid search.
    """
    parser = argparse.ArgumentParser(description="Run SimGNN.")
    parser.add_argument("--debug-dataset",
                        nargs="?",
                        default="/home/ouyangchao/binsimgnn/debug_heteroG_dataset",
	                help="Folder with debug_dataset")
    
    parser.add_argument("--dataset",
                        nargs="?",
                        default="/home/ouyangchao/binsimgnn/binkit_small_heteroG_dataset",
	                help="Folder with dataset")

    parser.add_argument("--epochs",
                        type=int,
                        default=20,
	                help="Number of training epochs.")

    parser.add_argument("--tensor-neurons",
                        type=int,
                        default=32,
	                help="Neurons in tensor network layer. Default is 16.") #NTN中W切片的数量

    parser.add_argument("--bottle-neck-neurons",
                        type=int,
                        default=32,
	                help="Bottle neck layer neurons. Default is 16.")

    parser.add_argument("--num_layers",
                    type=int,
                    default=3,
                help="GCN layers num.")


    parser.add_argument("--hidden_dim",
                    type=int,
                    default=32,
                help="hidden layer dimension")

    parser.add_argument("--heads",
                    type=int,
                    default=2,
                help="attention heads")
    
    parser.add_argument("--batch-pairs-size",  
                        type=int,
                        default=4,
	                help="一个batch中图对的个数,必须是偶数(正负对各一半)")

    parser.add_argument("--bins",
                        type=int,
                        default=32,
	                help="Similarity score bins. Default is 16.")

    parser.add_argument("--dropout",
                        type=float,
                        default=0.5,
	                help="Dropout probability. Default is 0.5.")

    parser.add_argument("--learning-rate",
                        type=float,
                        default=0.01,
	                help="initial Learning rate. ")

    parser.add_argument("--weight-decay",
                        type=float,
                        default=5*10**-4,
	                help="Adam weight decay")

    parser.add_argument("--save-dir",
                        type=str,
                        default=r'/home/ouyangchao/binsimgnn/binkit_small_model',
                        help="Dir to save the trained model")

    parser.add_argument("--load-path",
                        type=str,
                        default=None,
                        help="Load a pretrained model")

    return parser.parse_args()
