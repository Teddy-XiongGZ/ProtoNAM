import argparse
from config import Config
import pdb

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--data")
    parser.add_argument("--model")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_epoch", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--test_step", type=int, default=1)
    parser.add_argument("--p", type=int, default=1)
    parser.add_argument("--h_dim", type=int, default=64)
    parser.add_argument("--n_proto", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_layers_pred", type=int, default=2)
    parser.add_argument("--tau", type=float, default=16)
    parser.add_argument("--batch_norm", action=argparse.BooleanOptionalAction)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--dropout_output", type=float, default=0.0)
    parser.add_argument("--output_penalty", type=float, default=0.0)    
    parser.add_argument("--weight_decay", type=float, default=1e-8)
    parser.add_argument("--eval", action=argparse.BooleanOptionalAction)
    parser.add_argument("--exp_str", type=str, help="special string to identify an experiment")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    
    config = Config(
        data = args.data,
        model = args.model,
        lr = args.lr,
        max_epoch = args.max_epoch,
        batch_size = args.batch_size,
        test_step = args.test_step,
        p = args.p,
        h_dim = args.h_dim,
        n_proto = args.n_proto,
        n_layers = args.n_layers,
        n_layers_pred = args.n_layers_pred,
        tau = args.tau,
        batch_norm = False if args.batch_norm is None else True,
        dropout = args.dropout,
        dropout_output = args.dropout_output,
        output_penalty = args.output_penalty,
        weight_decay = args.weight_decay,
        eval = False if args.eval is None else True,
        exp_str = args.exp_str,
        device = args.device,
        fold = args.fold,
        seed = args.seed
    )
        
    config.train()