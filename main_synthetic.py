"""
VM: cpu, cpu, mem, mem, cpu % 16, cpu % 16  (0 is full, 1 is empty)
PM: cpu, cpu, mem, mem, fragment_rate, cpu % 16, fragment_rate, cpu % 16
cpu % 16 = round(normalized_cpu * 88) % 16 / 16
fragment_rate = round(normalized_cpu * 88) % 16 / round(normalized_cpu * 88)
To rescale memory, mem * 368776
"""

import argparse
from copy import deepcopy
import os
import random
from distutils.util import strtobool

import wandb

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from envs.linemsg import parallel_env

import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-cycles", type=int, default=125, help="maximum number of redeploy steps")
    parser.add_argument("--actor-lr", type=float, default=1e-3,
                        help="the learning rate of the optimizer")
    parser.add_argument("--critic-lr", type=float, default=1e-3,
                        help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=1,
                        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=400,
                        help="total timesteps of the experiments")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--track", action='store_true',
                        help="if toggled, this experiment will be tracked with Weights and Biases")

    # Algorithm specific arguments
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--target-network-frequency", type=int, default=1,
                        help="how often to update the target Q networks")
    parser.add_argument("--tau", type=float, default=0.005,
                        help="how much to update the target Q networks each time")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="the discount factor gamma")
    parser.add_argument("--update-epochs", type=int, default=1,
                        help="the K epochs to update the Q networks")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="the maximum norm for the gradient clipping")
    args = parser.parse_args()
    return args


class Agent(nn.Module):
    def __init__(self, num_states, num_actions):
        super().__init__()
        self.state_layer1 = nn.Embedding(num_states + 1, 4)
        self.dropout1 = nn.Dropout(p=0.1)
        self.state_layer2 = self._layer_init(nn.Linear(4 * (1 + 2 * n_obs_neighbors), 32))
        self.actor = self._layer_init(nn.Linear(32, num_actions), std=0.01)

    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def get_action(self, state, action=None):
        state_latent = self.dropout1(torch.flatten(self.state_layer1(state.long()), start_dim=1))
        logits = self.actor(torch.tanh(self.state_layer2(torch.tanh(state_latent))))
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy()


class Critic(nn.Module):
    def __init__(self, num_states, num_actions, n_obs_neighbors):
        super().__init__()

        self.state_layer1 = nn.Embedding(num_states + 1, 4)
        self.action_layer1 = nn.Embedding(num_actions + 1, 4)
        self.dropout1 = nn.Dropout(0.1)
        self.critic = self._layer_init(nn.Linear(4 * (1 + 2 * n_obs_neighbors) * 2, 32))
        self.critic2 = self._layer_init(nn.Linear(32, 1), std=1.0)
        self.num_actions = num_actions
        self.n_obs_neighbors = n_obs_neighbors

    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def get_value(self, state, action):
        state_latent = torch.flatten(self.state_layer1(state.long()), start_dim=1)
        action_latent = torch.flatten(self.action_layer1(action), start_dim=1)
        return self.critic2(torch.tanh(self.critic(torch.tanh(self.dropout1(torch.cat([state_latent, action_latent], dim=-1))))))
        # return self.critic(torch.cat([state_latent, action_latent], dim=-1))


if __name__ == "__main__":
    args = parse_args()
    save_every_step = 50
    plot_every_step = 20
    test_every_step = 20
    gamma = 0.99
    max_cycles = args.max_cycles

    dual_mu = 0.1
    eta_mu = 10
    entropy_constraint = 3
    n_sample_traj = 5
    n_sample_Q_traj = 4
    args.minibatch_size = 512
    n_obs_neighbors = 3
    n_sample_Q_steps = n_sample_Q_traj * max_cycles
    run_name = f"synthetic_N{n_obs_neighbors}_{dual_mu}_{eta_mu}_{entropy_constraint}_{args.seed}" \
               f"_{utils.name_with_datetime()}"
    np.set_printoptions(precision=4)
    np.set_printoptions(suppress=True)
    torch.backends.cudnn.benchmark = True
    torch.set_default_dtype(torch.float32)

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    # env setup
    env = parallel_env(
        render_mode="human", n_obs_neighbors=n_obs_neighbors, num_agents=5, max_iter=max_cycles
    )
    num_agents = len(env.possible_agents)
    # agents 0...num_agents are ordered from left to right
    num_states = 2
    num_actions = 2
    obs_size = n_obs_neighbors * 2 + 1
    n_piston_positions = 2  # env.unwrapped.n_piston_positions

    if args.track:
        wandb_tags = []
        wandb.init(entity="ANONYMOUS",
                   project="SafeRL",
                   name=run_name,
                   sync_tensorboard=True,
                   monitor_gym=True, config={
                'model': __file__[:__file__.find('.py')],
                'actor_lr': args.actor_lr,
                'critic_lr': args.critic_lr,
                'gamma': args.gamma,
                'dual_mu': dual_mu,
                'eta_mu': eta_mu,
                'max_cycles': max_cycles,
                'entropy_constraint': entropy_constraint,
                'n_sample_traj': n_sample_traj,
                'n_sample_Q_traj': n_sample_Q_traj,
                'n_obs_neighbors': n_obs_neighbors,
                'minibatch_size': args.minibatch_size,
                'update_epochs': args.update_epochs,
                'max_grad_norm': args.max_grad_norm},
                   # notes="",
                   tags=wandb_tags,
                   save_code=True
                   )
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )

    actors = [Agent(num_states, num_actions) for i in range(num_agents)]
    critics1 = [Critic(num_states, num_actions, n_obs_neighbors) for i in range(num_agents)]
    critics1_target = deepcopy(critics1)
    critics1_copy = [Critic(num_states, num_actions, n_obs_neighbors) for i in range(num_agents)]
    critics1_copy_target = deepcopy(critics1_copy)
    critics2 = [Critic(num_states, num_actions, n_obs_neighbors) for i in range(num_agents)]
    critics2_target = deepcopy(critics2)
    critics2_copy = [Critic(num_states, num_actions, n_obs_neighbors) for i in range(num_agents)]
    critics2_copy_target = deepcopy(critics2_copy)
    actor_optimizer = [optim.Adam(x.parameters(),
                                  lr=args.actor_lr) for x in actors]
    critic1_optimizer = [optim.Adam(x.parameters(),
                                    lr=args.critic_lr) for x in critics1]
    critic1_copy_optimizer = [optim.Adam(x.parameters(),
                                         lr=args.critic_lr) for x in critics1_copy]
    critic2_optimizer = [optim.Adam(x.parameters(),
                                    lr=args.critic_lr) for x in critics2]
    critic2_copy_optimizer = [optim.Adam(x.parameters(),
                                         lr=args.critic_lr) for x in critics2_copy]

    for x in actors:
        x.to(device)
    for x in critics1:
        x.to(device)
    for x in critics1_copy:
        x.to(device)
    for x in critics2:
        x.to(device)
    for x in critics2_copy:
        x.to(device)
    for x in critics1_target:
        x.to(device)
        for p in x.parameters():
            p.requires_grad = False
    for x in critics1_copy_target:
        x.to(device)
        for p in x.parameters():
            p.requires_grad = False
    for x in critics2_target:
        x.to(device)
        for p in x.parameters():
            p.requires_grad = False
    for x in critics2_copy_target:
        x.to(device)
        for p in x.parameters():
            p.requires_grad = False

    # ALGO Logic: Storage setup
    rb_obs = torch.zeros(n_sample_traj, max_cycles, num_agents, obs_size).to(device)
    rb_obs_next = torch.zeros(n_sample_traj, max_cycles, num_agents, obs_size).to(device)
    rb_actions = torch.zeros(n_sample_traj, max_cycles, num_agents, dtype=torch.int64).to(device)
    rb_rewards = torch.zeros(n_sample_traj, max_cycles, num_agents).to(device)
    rb_terms = torch.zeros(n_sample_traj, max_cycles, num_agents).to(device)
    rb_pos_y = torch.zeros(n_sample_traj, max_cycles, num_agents, n_piston_positions).to(device)
    rb_occupancy_measure = torch.zeros(n_sample_traj, num_agents, n_piston_positions).to(device)
    rb_end_steps = torch.zeros(n_sample_traj, dtype=torch.long).to(device)

    rb_q_obs = torch.zeros(n_sample_Q_steps, num_agents, obs_size).to(device)
    rb_q_obs_next = torch.zeros(n_sample_Q_steps, num_agents, obs_size).to(device)
    rb_q_actions = torch.zeros(n_sample_Q_steps, num_agents, dtype=torch.int64).to(device)
    rb_q_rewards1 = torch.zeros(n_sample_Q_steps, num_agents).to(device)
    rb_q_rewards2 = torch.zeros(n_sample_Q_steps, num_agents).to(device)
    rb_q_terms = torch.zeros(n_sample_Q_steps, num_agents).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0

    num_updates = args.total_timesteps
    pbar = trange(1, num_updates + 1)
    for update in pbar:
        total_episodic_return = torch.zeros(n_sample_traj, num_agents).to(device)

        # Step 3-5: collect n_sample_traj trajectories and find average occupancy measure
        # env.unwrapped.set_initial_ball_state()  # TODO: check if reset is successful
        for traj_id in range(n_sample_traj):
            next_obs = env.reset()

            for step in range(0, max_cycles):
                global_step += 1

                obs = utils.batchify(next_obs, device)

                with torch.no_grad():
                    # get action from the agent
                    actions = torch.zeros(num_agents, dtype=torch.int64, device=device)
                    for agent_id in range(num_agents):
                        agent_action, agent_logprob, _ = actors[agent_id].get_action(obs[agent_id].unsqueeze(0))
                        actions[agent_id] = agent_action

                # execute the environment and log data
                next_obs, rewards, terms, truncs, infos = env.step(
                    utils.unbatchify(actions, env)
                )

                # add to episode storage
                rb_obs[traj_id, step] = obs
                rb_rewards[traj_id, step] = utils.batchify(rewards, device)
                rb_actions[traj_id, step] = actions
                rb_terms[traj_id, step] = utils.batchify(terms, device)

                # compute episodic return
                total_episodic_return[traj_id] += rb_rewards[traj_id, step]
                rb_end_steps[traj_id] = step

                rb_pos_y[traj_id, step] = F.one_hot(utils.batchify(infos, device, 'pos_y').to(torch.int64),
                                                    num_classes=n_piston_positions)

                # if we reach termination or truncation, end
                if any([terms[a] for a in terms]) or any([truncs[a] for a in truncs]):
                    break

            # bootstrap value if not done
            with torch.no_grad():
                rb_occupancy_measure[traj_id] = rb_pos_y[traj_id, rb_end_steps[traj_id]]
                for t in reversed(range(rb_end_steps[traj_id])):
                    rb_occupancy_measure[traj_id] = rb_pos_y[traj_id, t] \
                                                    + args.gamma * rb_occupancy_measure[traj_id]

        avg_occupancy_measure = torch.mean(rb_occupancy_measure, dim=0)

        # flatten the batch
        keep_mask = torch.arange(max_cycles).unsqueeze(0).repeat(n_sample_traj, 1).to(device)
        keep_mask = keep_mask <= rb_end_steps.unsqueeze(-1)
        b_obs = rb_obs[keep_mask]
        b_obs_next = rb_obs_next[keep_mask]
        b_actions = rb_actions[keep_mask]
        b_rewards = rb_rewards[keep_mask]
        b_terms = rb_terms[keep_mask]

        avg_episodic_return = total_episodic_return.mean().item()
        if args.track:
            writer.add_scalar("episode_details/episodic_return", avg_episodic_return, global_step)
            writer.add_scalar("episode_details/end_step", rb_end_steps.to(torch.float32).mean().item(), global_step)
        pbar.set_description(f'Total return: {avg_episodic_return:.2f}')


        # Step 6: collect n_sample_Q_traj trajectories to update the value functions
        # def r_f(is_right):
        #     return is_right

        def r_mu(pos_y):  # entropy >= c
            select_occupancy = avg_occupancy_measure[torch.arange(pos_y.shape[0], device=device), pos_y]
            return -(1 - gamma) * (torch.log(select_occupancy * (1 - gamma) + 1e-6) + 1) * 500


        # rollout for data collection
        with torch.no_grad():
            reset = True
            for q_step in range(n_sample_Q_steps):
                if reset:
                    next_obs = env.reset()
                    reset = False
                obs = utils.batchify(next_obs, device)

                # get action from the agent
                actions = torch.zeros(num_agents, dtype=torch.int64, device=device)
                for agent_id in range(num_agents):
                    agent_action, agent_logprob, _ = actors[agent_id].get_action(obs[agent_id].unsqueeze(0))
                    actions[agent_id] = agent_action

                next_obs, rewards, terms, truncs, infos = env.step(
                    utils.unbatchify(actions, env)
                )

                if len(next_obs) == 0:
                    reset = True
                    q_step -= 1
                else:
                    rb_q_obs[q_step] = obs
                    rb_q_obs_next[q_step] = utils.batchify(next_obs, device)
                    rb_q_actions[q_step] = actions
                    rb_q_rewards1[q_step] = utils.batchify(rewards, device)
                    # print('info: ', infos)
                    # print('r_mu: ', r_mu(utils.batchify(infos, device, 'pos_y').to(torch.int64)))
                    rb_q_rewards2[q_step] = r_mu(utils.batchify(infos, device, 'pos_y').to(torch.int64))
                    rb_q_terms[q_step] = utils.batchify(terms, device)

        if args.track:
            writer.add_scalar("episode_details/avg_reward1", rb_q_rewards1.mean().item(), global_step)
            writer.add_scalar("episode_details/avg_reward2", rb_q_rewards2.mean().item(), global_step)
        pbar.set_description(f'Total return: {avg_episodic_return:.2f}')

        # optimize the value networks
        b_inds = np.arange(n_sample_Q_steps)
        for agent_id in range(num_agents):
            critic1_loss_track = []
            critic2_loss_track = []
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for index, start in enumerate(range(0, n_sample_Q_steps, args.minibatch_size)):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]

                    # get critic_values
                    neighbor_actions = rb_q_actions[mb_inds, max(0, agent_id - n_obs_neighbors):
                                                             min(agent_id + n_obs_neighbors + 1, num_agents)]
                    if agent_id - n_obs_neighbors < 0:
                        neighbor_actions = torch.cat([torch.zeros(mb_inds.shape[0], n_obs_neighbors - agent_id,
                                                                  dtype=torch.int64, device=device) + 2,
                                                      neighbor_actions], dim=1)

                    if agent_id + n_obs_neighbors + 1 > num_agents:
                        neighbor_actions = torch.cat([neighbor_actions,
                                                      torch.zeros(mb_inds.shape[0],
                                                                  n_obs_neighbors + agent_id + 1 - num_agents,
                                                                  dtype=torch.int64, device=device) + 2], dim=1)
                    assert neighbor_actions.shape[1] == 2 * n_obs_neighbors + 1

                    neighbor_next_actions = torch.zeros(mb_inds.shape[0], 2 * n_obs_neighbors + 1,
                                                        dtype=torch.int64, device=device) + 2
                    for fill_id, neighbor_id in enumerate(range(agent_id - n_obs_neighbors,
                                                                agent_id + n_obs_neighbors + 1)):
                        if 0 <= neighbor_id < num_agents:
                            neighbor_next_actions[:, fill_id], _, _ \
                                = actors[neighbor_id].get_action(rb_q_obs_next[mb_inds, neighbor_id])
                    # TODO: consider adding target_actor
                    critic1_current_score = critics1[agent_id].get_value(rb_q_obs[mb_inds, agent_id],
                                                                         neighbor_actions).squeeze()
                    critic1_copy_current_score = critics1_copy[agent_id].get_value(rb_q_obs[mb_inds, agent_id],
                                                                                   neighbor_actions).squeeze()
                    critic1_next_score = critics1_target[agent_id].get_value(rb_q_obs_next[mb_inds, agent_id],
                                                                             neighbor_next_actions).squeeze()
                    critic1_copy_next_score = critics1_copy_target[agent_id].get_value(rb_q_obs_next[
                                                                                           mb_inds, agent_id],
                                                                                       neighbor_next_actions).squeeze()
                    critic1_final_score = torch.minimum(critic1_next_score, critic1_copy_next_score)
                    critic1_target_score = (1 - rb_q_terms[mb_inds, agent_id]) * gamma * critic1_final_score + \
                                           rb_q_rewards1[mb_inds, agent_id]
                    # print('agent_id: ', agent_id)
                    # print('rewards1: ', rb_q_rewards1[mb_inds, agent_id][:10])
                    # print(f'critic1 current: {critic1_current_score[:10]}, target: {critic1_target_score[:10]}')
                    loss_critic1 = F.mse_loss(critic1_current_score, critic1_target_score.detach())
                    critic1_optimizer[agent_id].zero_grad(set_to_none=True)
                    loss_critic1.backward()
                    critic1_optimizer[agent_id].step()
                    loss_critic1_copy = F.mse_loss(critic1_copy_current_score, critic1_target_score.detach())
                    critic1_copy_optimizer[agent_id].zero_grad(set_to_none=True)
                    loss_critic1_copy.backward()
                    critic1_copy_optimizer[agent_id].step()
                    critic1_loss_track.append(loss_critic1.item())

                    critic2_current_score = critics2[agent_id].get_value(rb_q_obs[mb_inds, agent_id],
                                                                         neighbor_actions).squeeze()
                    critic2_copy_current_score = critics2_copy[agent_id].get_value(rb_q_obs[mb_inds, agent_id],
                                                                                   neighbor_actions).squeeze()
                    critic2_next_score = critics2_target[agent_id].get_value(rb_q_obs_next[mb_inds, agent_id],
                                                                             neighbor_next_actions).squeeze()
                    critic2_copy_next_score = critics2_copy_target[agent_id].get_value(rb_q_obs_next[
                                                                                           mb_inds, agent_id],
                                                                                       neighbor_next_actions).squeeze()
                    critic2_final_score = torch.minimum(critic2_next_score, critic2_copy_next_score)
                    critic2_target_score = (1 - rb_q_terms[mb_inds, agent_id]) * gamma * critic2_final_score + \
                                           rb_q_rewards2[mb_inds, agent_id]
                    # print('rb_q_terms: ', rb_q_terms[mb_inds, agent_id][:10])
                    # print(f'critic2 current: {critic2_current_score[:10]}, target: {critic2_target_score[:10]}')

                    loss_critic2 = F.mse_loss(critic2_current_score, critic2_target_score.detach())
                    critic2_optimizer[agent_id].zero_grad()
                    loss_critic2.backward()
                    critic2_optimizer[agent_id].step()
                    loss_critic2_copy = F.mse_loss(critic2_copy_current_score, critic2_target_score.detach())
                    critic2_copy_optimizer[agent_id].zero_grad()
                    loss_critic2_copy.backward()
                    critic2_copy_optimizer[agent_id].step()
                    critic2_loss_track.append(loss_critic2.item())

                # update the target networks
                if epoch % args.target_network_frequency == 0:
                    for param, target_param in zip(critics1[agent_id].parameters(),
                                                   critics1_target[agent_id].parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                    for param, target_param in zip(critics1_copy[agent_id].parameters(),
                                                   critics1_copy_target[agent_id].parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                    for param, target_param in zip(critics2[agent_id].parameters(),
                                                   critics2_target[agent_id].parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                    for param, target_param in zip(critics2_copy[agent_id].parameters(),
                                                   critics2_copy_target[agent_id].parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if args.track and (agent_id + 1) % 5 == 0:
                # TRY NOT TO MODIFY: record rewards for plotting purposes
                writer.add_scalar(f"losses/agent_{agent_id}_critic1_loss",
                                  sum(critic1_loss_track) / len(critic1_loss_track), global_step)
                writer.add_scalar(f"losses/agent_{agent_id}_critic2_loss",
                                  sum(critic2_loss_track) / len(critic2_loss_track), global_step)

        # step 7: Lagrangian multiplier update
        discounted_occupancy = ((1 - gamma) * avg_occupancy_measure)
        occupancy_entropy = torch.sum(discounted_occupancy * torch.log(discounted_occupancy + 1e-6)) + entropy_constraint
        dual_mu = max(0, eta_mu * occupancy_entropy)
        constraint_violation = torch.sum(torch.clamp(discounted_occupancy * torch.log(discounted_occupancy + 1e-6) + entropy_constraint, min=0))

        if args.track:
            # TRY NOT TO MODIFY: record rewards for plotting purposes
            writer.add_scalar(f"episode_details/dual_mu", dual_mu, global_step)
            writer.add_scalar(f"episode_details/occupancy_entropy", occupancy_entropy, global_step)
            writer.add_scalar(f"episode_details/constraint_violation", constraint_violation, global_step)

        # step 8-9: policy gradient evaluation & parameter update
        policy_bsz = b_obs.shape[0]
        for agent_id in range(num_agents):
            with torch.no_grad():
                neighbor_score_sum = torch.zeros(policy_bsz, device=device)
                critic1_score_tracker = []
                critic2_score_tracker = []
                for neighbor_id in range(max(0, agent_id - n_obs_neighbors),
                                         min(agent_id + n_obs_neighbors + 1, num_agents)):
                    neighbor_actions = b_actions[:, max(0, agent_id - n_obs_neighbors):
                                                    min(agent_id + n_obs_neighbors + 1, num_agents)]
                    if agent_id - n_obs_neighbors < 0:
                        neighbor_actions = torch.cat([torch.zeros(policy_bsz, n_obs_neighbors - agent_id,
                                                                  dtype=torch.int64, device=device) + 2,
                                                      neighbor_actions], dim=1)

                    if agent_id + n_obs_neighbors + 1 > num_agents:
                        neighbor_actions = torch.cat([neighbor_actions,
                                                      torch.zeros(policy_bsz,
                                                                  n_obs_neighbors + agent_id + 1 - num_agents,
                                                                  dtype=torch.int64, device=device) + 2], dim=1)
                    assert neighbor_actions.shape[1] == 2 * n_obs_neighbors + 1

                    critic1_neighbor_score = critics1[neighbor_id].get_value(b_obs[:, agent_id],
                                                                             neighbor_actions).squeeze()
                    critic1_copy_neighbor_score = critics1_copy[neighbor_id].get_value(b_obs[:, agent_id],
                                                                                       neighbor_actions).squeeze()
                    critic2_neighbor_score = critics2[neighbor_id].get_value(b_obs[:, agent_id],
                                                                             neighbor_actions).squeeze()
                    critic2_copy_neighbor_score = critics2_copy[neighbor_id].get_value(b_obs[:, agent_id],
                                                                                       neighbor_actions).squeeze()
                    critic1_final_score = torch.minimum(critic1_neighbor_score, critic1_copy_neighbor_score)
                    critic2_final_score = torch.minimum(critic2_neighbor_score, critic2_copy_neighbor_score)

                    neighbor_critic_score = critic1_final_score + dual_mu * critic2_final_score
                    neighbor_score_sum = neighbor_score_sum + neighbor_critic_score
                    critic1_score_tracker.append(torch.mean(critic1_neighbor_score).item())
                    critic2_score_tracker.append(torch.mean(critic2_neighbor_score).item())

                average_score = neighbor_score_sum / num_agents
                average_critic1_score = sum(critic1_score_tracker) / len(critic1_score_tracker)
                average_critic2_score = sum(critic2_score_tracker) / len(critic2_score_tracker)

            _, agent_logprob, _ = actors[agent_id].get_action(b_obs[:, agent_id], action=b_actions[:, agent_id])
            pg_loss = torch.mean(-average_score.detach() * agent_logprob)
            actor_optimizer[agent_id].zero_grad(set_to_none=True)
            pg_loss.backward()
            nn.utils.clip_grad_norm_(actors[agent_id].parameters(), args.max_grad_norm)
            actor_optimizer[agent_id].step()

            if args.track and (agent_id + 1) % 5 == 0:
                # TRY NOT TO MODIFY: record rewards for plotting purposes
                writer.add_scalar(f"losses/agent_{agent_id}_policy_loss", pg_loss.item(), global_step)
                writer.add_scalar(f"losses/agent_{agent_id}_critic1_score", average_critic1_score, global_step)
                writer.add_scalar(f"losses/agent_{agent_id}_critic2_score", average_critic2_score, global_step)

            """
            if (update + 1) % save_every_step == 0:
                utils.save_checkpoint({'global_step': global_step,
                                       'state_dict': agent.state_dict(),
                                       'optim_dict': optim.state_dict()},
                                      global_step=global_step,
                                      checkpoint=f"runs/{run_name}")
            """

        if args.track and (update + 1) % test_every_step == 0:
            with torch.no_grad():
                for sample_id in range(5):
                    print('=' * 10 + f" sample {sample_id} " + '=' * 10)
                    total_test_episode_return = torch.zeros(num_agents, device=device)
                    next_obs = env.reset()

                    for step in range(0, max_cycles):
                        state_str = env.render()
                        print(f'At step {step}: \n')
                        print(state_str)
                        obs = utils.batchify(next_obs, device)

                        with torch.no_grad():
                            # get action from the agent
                            actions = torch.zeros(num_agents, dtype=torch.int64, device=device)
                            for agent_id in range(num_agents):
                                agent_action, _, _ = actors[agent_id].get_action(obs[agent_id].unsqueeze(0))
                                actions[agent_id] = agent_action

                        # execute the environment and log data
                        next_obs, rewards, terms, truncs, infos = env.step(
                            utils.unbatchify(actions, env)
                        )

                        # compute episodic return
                        total_test_episode_return += utils.batchify(rewards, device)

                        # if we reach termination or truncation, end
                        if any([terms[a] for a in terms]) or any([truncs[a] for a in truncs]):
                            break

    env.close()
    if args.track:
        writer.close()
