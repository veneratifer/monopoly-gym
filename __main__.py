from model.trainer import Trainer
import numpy as np
from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser(prog='Monopoly Gym')
    parser.add_argument('-n', '--num_episodes', default=1, type=int)
    parser.add_argument('-o', '--out_path', default='./model.pt')
    parser.add_argument('-e', '--eval', action="store_true")
    parser.add_argument('-v', '--vis', action="store_true")
    parser.add_argument('-p', '--model_path')
    parser.add_argument('-s', '--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    trainer = Trainer()
    if args.eval:
        assert args.model_path != None, "cannot run evaluation without specifing --model_path"
        trainer.eval(args.model_path, args.num_episodes, args.vis)
    else:
        trainer.train(num_episodes=args.num_episodes, out_path=args.out_path)