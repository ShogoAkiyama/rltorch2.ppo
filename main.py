import os
import argparse
import yaml
import torch
from datetime import datetime
from rltorch2_ppo.env.procgen_env_wrapper import ProcgenEnvWrapper
from rltorch2_ppo.env.my_env import MyEnv

from rltorch2_ppo.agent import PPO


def main(args):
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Make vectorized environment.
    envs = ProcgenEnvWrapper()
    # envs = MyEnv()

    # Specify the directory to log.
    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        args.log_dir,
        args.env_id.split('NoFrameskip')[0],
        f'PPO-{args.seed}-{time}')

    # PPO agent.
    agent = PPO(
        envs=envs, device=device, log_dir=log_dir,
        num_processes=args.num_processes, **config)
    agent.run().save_models(filename='final_model.pth')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--config', type=str, default='config/ppo.yaml')
    parser.add_argument('--env_id', default='procgen:procgen-coinrun-v0')
    parser.add_argument('--num_processes', type=int, default=8)
    parser.add_argument('--log_dir', default='logs/ppo')
    parser.add_argument('--seed', type=int, default=0)
    main(parser.parse_args())
