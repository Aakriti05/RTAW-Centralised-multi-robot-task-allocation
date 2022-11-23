import argparse
import functools

import gym
import warehouse
import gym.spaces
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
# from train_agent_with_evaluation import train_agent_with_evaluation

from network_policy import Network_policy
from network_value import Network_value
from torch import nn

import pfrl
from pfrl import experiments, utils
from pfrl.agents import PPO
from pfrl.policies import SoftmaxCategoricalHead


def main():
    import logging
    writer = SummaryWriter()

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=1, help="GPU to use, set to -1 if no GPU.")
    parser.add_argument("--env", type=str, default="warehouse-v0")
    parser.add_argument("--num-envs", type=int, default=4, help="Number of envs run in parallel.")
    parser.add_argument("--seed", type=int, default=6, help="Random seed [0, 2 ** 32)")
    parser.add_argument("--outdir", type=str, default="results", help=("Directory path to save output files."
            " If it does not exist, it will be created."),)
    # parser.add_argument("--logs", type=str, default="logs", help=("Directory path to log files."),)
    parser.add_argument("--steps", type=int, default=2 * 10 ** 6, help="Total number of timesteps to train the agent.",)
    parser.add_argument("--eval-interval", type=int, default=2000, help="Interval in timesteps between evaluations.",)
    parser.add_argument("--eval-n-runs", type=int, default=20, help="Number of episodes run for each evaluation.",)
    parser.add_argument("--render", action="store_true", help="Render env states in a GUI window.")
    parser.add_argument("--demo", action="store_true", help="Just run evaluation, not training.")
    parser.add_argument("--load-pretrained", action="store_true", default=False)
    parser.add_argument("--load", type=str, default="", help="Directory to load agent from.")
    parser.add_argument("--log-level", type=int, default=logging.INFO, help="Level of the root logger.")
    parser.add_argument("--monitor", action="store_true", help="Wrap env with gym.wrappers.Monitor.")
    parser.add_argument("--log-interval", type=int, default=800, help="Interval in timesteps between outputting log messages during training")
    parser.add_argument("--update-interval", type=int, default=512, help="Interval in timesteps between model updates.",)
    parser.add_argument("--epochs", type=int, default=16, help="Number of epochs to update model for per PPO iteration.",)
    parser.add_argument("--batch-size", type=int, default=32, help="Minibatch size")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)

    # Set a random seed used in PFRL
    # utils.set_random_seed(args.seed)
    process_seeds = np.arange(args.num_envs) + args.seed * args.num_envs
    assert process_seeds.max() < 2 ** 32

    args.outdir = experiments.prepare_output_dir(args, args.outdir)
    
    def make_env(process_idx, test):
        env = gym.make(args.env)
        # Use different random seeds for train and test envs
        process_seed = int(process_seeds[process_idx])
        env_seed = 2 ** 32 - 1 - process_seed if test else process_seed
        env.seed(env_seed)
        # Cast observations to float32 because our model uses float32
        env = pfrl.wrappers.CastObservationToFloat32(env)
        if args.monitor:
            env = pfrl.wrappers.Monitor(env, args.outdir)
        if args.render:
            env = pfrl.wrappers.Render(env)
        return env

    def make_batch_env(test):
        return pfrl.envs.MultiprocessVectorEnv(
            [
                functools.partial(make_env, idx, test)
                for idx, env in enumerate(range(args.num_envs))
            ]
        )


    # Only for getting timesteps, and obs-action spaces
    sample_env = gym.make(args.env)
    timestep_limit = 600 # sample_env.spec.max_episode_steps
    print(timestep_limit)
    obs_space = sample_env.observation_space
    action_space = sample_env.action_space
    print("Observation space:", obs_space)
    print("Action space:", action_space)

    assert isinstance(action_space, gym.spaces.Box)

    # Normalize observations based on their empirical mean and variance
    obs_normalizer = pfrl.nn.EmpiricalNormalization(obs_space.low.size, clip_threshold=5)
    # obs_size = obs_space.low.size
    action_size = action_space.low.size

    # train a smaller network
    # policy = torch.nn.Sequential(
    #     nn.Linear(41, 64),
    #     nn.ReLU(),
    #     nn.Linear(64, 128),
    #     nn.ReLU(),
    #     nn.Linear(128, 128),
    #     nn.ReLU(),
    #     nn.Linear(128, 32),
    #     nn.ReLU(),
    #     nn.Linear(32, 2),
    #     # torch.argmax(),
    #     # SoftmaxCategoricalHead(),
    #     # pfrl.policies.GaussianHeadWithStateIndependentCovariance(
    #     #     action_size=action_size,
    #     #     var_type="diagonal",
    #     #     var_func=lambda x: torch.exp(2 * x),  # Parameterize log std
    #     #     var_param_init=0,  # log std = 0 => std = 1
    # )


    policy = Network_policy()

    # vf = torch.nn.Sequential(
    #     nn.Linear(41, 64),
    #     nn.ReLU(),
    #     nn.Linear(64, 128),
    #     nn.ReLU(),
    #     nn.Linear(128, 128),
    #     nn.ReLU(),
    #     nn.Linear(128, 32),
    #     nn.ReLU(),
    #     nn.Linear(32, 1),
    # )

    vf = torch.nn.Sequential(
        Network_value(),
        nn.Linear(10, 1),        #6
    )

    # linearx = nn.Linear(128, 100)
    # pfrl.initializers.init_lecun_normal(linearx.weight, 1e-2)


    # While the original paper initialized weights by normal distribution,
    # we use orthogonal initialization as the latest openai/baselines does.
    def ortho_init(layer, gain):
        nn.init.orthogonal_(layer.weight, gain=gain)
        nn.init.zeros_(layer.bias)

    # ortho_init(policy[0], gain=1)
    # ortho_init(policy[2], gain=1)
    # ortho_init(policy[4], gain=1)
    # ortho_init(policy[6], gain=1)
    # ortho_init(policy[8], gain=1e-2)

    # ortho_init(vf[0], gain=1)
    # ortho_init(vf[2], gain=1)
    # ortho_init(vf[4], gain=1)
    # ortho_init(vf[6], gain=1)
    ortho_init(vf[1], gain=1)

    # Combine a policy and a value function into a single model
    model = pfrl.nn.Branched(policy, vf)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-5)

    agent = PPO(
        model,
        opt,
        obs_normalizer=obs_normalizer,
        gpu=args.gpu,
        value_func_coef=0.0002,
        update_interval=args.update_interval,
        minibatch_size=args.batch_size,
        epochs=args.epochs,
        clip_eps_vf=None,
        entropy_coef=0.01,
        standardize_advantages=True,
        gamma=0.99,
        lambd=0.95,
    )

    # print(model)
    

    if args.load or args.load_pretrained:    
        # either load or load_pretrained must be false
        # print("loading")
        assert not args.load or not args.load_pretrained
        if args.load:
            agent.load(args.load)
        else:
            agent.load(utils.download_model("PPO", args.env, model_type="final")[0])

        temp_policy = agent.model.child_modules[0]
        print(temp_policy)


    if args.demo:
        env = gym.make(args.env)
        # env = make_batch_env(True)
        state = env.reset()
        print("entered")
        

        for i in range(500):
            # print(state)
            state = torch.Tensor([state])
            # print("state", state)
            temp_policy = temp_policy.cuda()
            state = state.cuda()

            output = temp_policy(state)
            # print(output.shape, output)
            # action= agent.act(state)
            action = torch.argmax(output, dim=-1)
            action_inp = action.cpu().detach().numpy() 
            # print("action", action_inp.shape, " ", action_inp)

            state, reward, done, x = env.step(action_inp)

        # env = make_batch_env(True)
        # eval_stats = experiments.eval_performance(
        #     env=env,
        #     agent=agent,
        #     n_steps=None,
        #     n_episodes=args.eval_n_runs,
        #     max_episode_len=timestep_limit,
        # )
        # print(
        #     "n_runs: {} mean: {} median: {} stdev {}".format(
        #         args.eval_n_runs,
        #         eval_stats["mean"],
        #         eval_stats["median"],
        #         eval_stats["stdev"],
        #     )
        # )
        # import json
        # import os

        # with open(os.path.join(args.outdir, "demo_scores.json"), "w") as f:
        #     json.dump(eval_stats, f)

    else:
        # experiments.train_agent_with_evaluation(
        #     agent=agent,
        #     env=make_env(0, False),
        #     eval_env=make_env(0, True),
        #     outdir=args.outdir,
        #     steps=args.steps,
        #     eval_n_steps=None,
        #     eval_n_episodes=args.eval_n_runs,
        #     eval_interval=args.eval_interval,
        #     # log_interval=args.log_interval,
        #     train_max_episode_len=timestep_limit,
        #     eval_max_episode_len=timestep_limit,
        #     save_best_so_far_agent=True,
        #     use_tensorboard = True,
        # )

        experiments.train_agent_batch_with_evaluation(
            agent=agent,
            env=make_batch_env(False),
            eval_env=make_batch_env(True),
            outdir=args.outdir,
            steps=args.steps,
            eval_n_steps=None,
            eval_n_episodes=args.eval_n_runs,
            eval_interval=args.eval_interval,
            log_interval=args.log_interval,
            eval_max_episode_len=timestep_limit,
            save_best_so_far_agent=True,
            use_tensorboard = True,
        )
    writer.close()


if __name__ == "__main__":
    main()
