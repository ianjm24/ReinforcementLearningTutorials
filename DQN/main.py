import argparse
from agent_dqn import Agent_DQN
from environment import Environment
from torch import tensor
import torch

seed = 11037


def parse():
    parser = argparse.ArgumentParser(description="runner")
    parser.add_argument('--env_name', default=None, help='environment name')
    parser.add_argument('--train_pg', action='store_true', help='whether train policy gradient')
    parser.add_argument('--train_dqn', action='store_true', help='whether train DQN')
    parser.add_argument('--test_pg', action='store_true', help='whether test policy gradient')
    parser.add_argument('--test_dqn', action='store_true', help='whether test DQN')
    parser.add_argument('--video_dir', default='videos', help='output video directory')
    parser.add_argument('--do_render', action='store_true', help='whether render environment')

    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--run_name', type=str, default='dqn_model', help='')
    parser.add_argument('--model_save_path', type=str, default='trained_models', help='')
    parser.add_argument('--model_save_interval', type=int, default=500, help='')
    parser.add_argument('--log_path', type=str, default='train_log.out', help='')
    parser.add_argument('--tensorboard_summary_path', type=str, default='tensorboard_summary', help='')
    parser.add_argument('--model_test_path', type=str, default='C:/Users/ianjm/OneDrive/Documents/GraduateClasses/2022Fall/ReinforcementLearning/Project/code/ReinforcementLearningTutorials/DQN/trained_models/dqn_model_120500.pt', help='')
    parser.add_argument('--metrics_capture_window', type=int, default=100, help='')
    parser.add_argument('--replay_size', type=int, default=10000, help='')
    parser.add_argument('--start_to_learn', type=int, default=5000, help='')
    parser.add_argument('--total_num_steps', type=int, default=5e7, help='')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='')
    parser.add_argument('--gamma', type=float, default=0.99, help='')
    parser.add_argument('--initial_epsilon', type=float, default=1.0, help='')
    parser.add_argument('--final_epsilon', type=float, default=0.005, help='')
    parser.add_argument('--steps_to_explore', type=int, default=1000000, help='')
    parser.add_argument('--network_update_interval', type=int, default=5000, help='')
    parser.add_argument('--episodes', type=int, default=50000000, help='')
    parser.add_argument('--network_train_interval', type=int, default=10, help='')
    parser.add_argument('--ddqn', type=bool, default=True, help='')
    torch.set_default_tensor_type('torch.cuda.FloatTensor' if torch.cuda.is_available() else 'torch.FloatTensor')

    args = parser.parse_args()
    return args


def run(args):
    if args.train_dqn:
        env_name = args.env_name or 'ALE/Breakout-v5'
        env = Environment(env_name, args, atari_wrapper=True)
        agent = Agent_DQN(env, args, seed)
        agent.train()

    if args.test_dqn:
        env = Environment('ALE/Breakout-v5', args, atari_wrapper=True, test=True)
        agent = Agent_DQN(env, args, seed)
        agent.test(total_episodes=100)

if __name__ == '__main__':
    args = parse()
    run(args)
