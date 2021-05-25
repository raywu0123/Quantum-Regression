from argparse import ArgumentParser


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('-n', '--n_qubit', type=int)
    parser.add_argument('-N', '--num_data', type=int, default=1000)
    parser.add_argument('-dist', type=str, default="Fidelity")
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--n_epoch', type=int, default=10)
    parser.add_argument('-m', '--model', type=str)
    parser.add_argument('-lr', type=float, default=1e-3)
    return parser
