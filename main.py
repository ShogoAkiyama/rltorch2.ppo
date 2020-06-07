import os
import argparse
import torch

from env.vector_env import make_vec_envs
from agent import PPO, utils


def main(args):
    # args = get_args()
    args.cuda = torch.cuda.is_available()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.log_dir, device, False)

    agent = PPO(
        envs, device,
        args.num_env_steps, args.num_steps, args.num_processes,
        args.recurrent_policy,
        args.clip_param, args.ppo_epoch, args.num_mini_batch,
        args.value_loss_coef, args.entropy_coef,
        lr=args.lr, gamma=args.gamma, gae_lambda=args.gae_lambda,
        eps=args.eps, max_grad_norm=args.max_grad_norm)

    agent.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--agent', default='ppo')
    parser.add_argument('--lr', type=float, default=2.5e-4)
    parser.add_argument('--eps', type=float, default=1e-5)
    parser.add_argument('--alpha', type=float, default=0.99)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--entropy-coef', type=float, default=0.01)
    parser.add_argument('--value-loss-coef', type=float, default=0.5)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--cuda-deterministic', action='store_true', default=False)
    parser.add_argument('--num-processes', type=int, default=8)
    parser.add_argument('--num-steps', type=int, default=128)
    parser.add_argument('--ppo-epoch', type=int, default=4)
    parser.add_argument('--num-mini-batch', type=int, default=4)
    parser.add_argument('--clip-param', type=float, default=0.1)
    parser.add_argument('--num-env-steps', type=int, default=10e6)
    parser.add_argument('--env-name', default='BreakoutNoFrameskip-v4')
    parser.add_argument('--log-dir', default='/tmp/gym/')
    parser.add_argument('--recurrent-policy', action='store_true', default=False)
    args = parser.parse_args()

    main(args)
