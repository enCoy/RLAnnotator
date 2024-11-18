from SCGLabelingRL.Models import Actor, Critic
from Memory import SequentialMemory
from torch.optim import Adam
from RLUtils import hard_update, to_tensor, soft_update, get_pretrained_cnn_output, get_pretrained_cnn_output_for_action
import numpy as np
import torch.nn as nn
import torch
from utils import *
import copy

criterion = nn.MSELoss()

class DDPG(object):
    def __init__(self, beat_length, n_states, n_actions, action_upper_range, pretrained_cnn, args):
        """

        :param n_states:
        :param n_actions:
        :param args: l_rate_actor: learning rate for actor
                              l_rate_critic: learning_rate for critic
                              memory_size: size of the memory of experiences
                              batch_size
                              tau: parameter for soft update of target between 0 and 1
                              discount: discount rate for the reward
                              epsilon
                              seed
        """
        if args.seed > 0:
            self.seed(args.seed)

        self.n_states = n_states
        self.other_state_dim = n_states - beat_length
        self.n_actions = n_actions
        self.action_upper_range = action_upper_range

        # create actor and critic networks
        net_cfg = {
            'hidden1': args.hidden1,
            'hidden2': args.hidden2,
        }
        self.beat_length = beat_length

        self.pretrained_cnn = pretrained_cnn
        if self.pretrained_cnn is not None:
            self.pretrained_cnn_target = copy.deepcopy(pretrained_cnn)
            self.pretrained_cnn_optim = Adam(self.pretrained_cnn.parameters(), lr=0.00003)
            # get the output shape of the pretrained cnn
            dummy_input = torch.randn(1, self.beat_length, 1)
            dummy_output = self.pretrained_cnn(dummy_input)
            self.pretrained_cnn_output_dim = dummy_output.shape[1] * dummy_output.shape[2]

        if self.pretrained_cnn is not None:
            self.actor = Actor(self.pretrained_cnn_output_dim + self.other_state_dim, self.n_actions, **net_cfg)
            self.actor_target = Actor(self.pretrained_cnn_output_dim + self.other_state_dim, self.n_actions, **net_cfg)
            self.critic = Critic(self.pretrained_cnn_output_dim + self.other_state_dim, self.n_actions, **net_cfg)
            self.critic_target = Critic(self.pretrained_cnn_output_dim + self.other_state_dim, self.n_actions, **net_cfg)
        else:
            self.actor = Actor(self.n_states, self.n_actions, **net_cfg)
            self.actor_target = Actor(self.n_states, self.n_actions, **net_cfg)
            self.critic = Critic(self.n_states, self.n_actions, **net_cfg)
            self.critic_target = Critic(self.n_states, self.n_actions, **net_cfg)

        self.actor_optim = Adam(self.actor.parameters(), lr=args.l_rate_actor)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.l_rate_critic)

        # make sure the target is with the same weight - IT is basically a COPY
        if self.pretrained_cnn is not None:
            hard_update(self.pretrained_cnn_target, self.pretrained_cnn)
        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

        # create replay buffer
        self.memory = SequentialMemory(limit=args.memsize, window_length=args.window_length)

        # hyperparams
        self.batch_size = args.batch_size
        self.tau = args.tau
        self.discount = args.discount
        self.depsilon = 1.0 / args.epsilon
        self.epsilon = 1.0
        self.s_t = None  # most recent state
        self.a_t = None  # most recent action
        self.is_training = True
        self.cuda()
        self.USE_CUDA = True

        self.a_noise_mean = args.action_noise_mean
        self.a_noise_std = args.action_noise_std


    def update_policy(self):
        # sample batch
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = self.memory.sample_and_split(self.batch_size)
        # volatile means no gradient backpropagation
        # prepare for target batch
        if self.pretrained_cnn is not None:
            next_state_batch = get_pretrained_cnn_output(next_state_batch, self.beat_length, self.pretrained_cnn_target)
            state_batch = get_pretrained_cnn_output(state_batch, self.beat_length, self.pretrained_cnn)

        next_q_values = self.critic_target([
            to_tensor(next_state_batch, volatile=True),   # states
            self.actor_target(to_tensor(next_state_batch, volatile=True))  # actions
        ])
        next_q_values.volatile = False

        target_q_batch = to_tensor(reward_batch) + self.discount * to_tensor(terminal_batch.astype(float)) * next_q_values

        # critic update
        self.critic.zero_grad()

        q_batch = self.critic([to_tensor(state_batch), to_tensor(action_batch)])

        value_loss = criterion(q_batch, target_q_batch)
        value_loss.backward()
        self.critic_optim.step()

        # actor update
        self.actor.zero_grad()
        policy_loss = -self.critic([
            to_tensor(state_batch),
            self.actor(to_tensor(state_batch))
        ])

        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()

        # target update
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)
        if self.pretrained_cnn is not None:
            soft_update(self.pretrained_cnn_target, self.pretrained_cnn, self.tau)

    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    def observe(self, r_t, s_t1, done):
        if self.is_training:
            self.memory.append(self.s_t, self.a_t, r_t, done)
            self.s_t = s_t1

    def select_action(self, s_t, decay_epsilon=True):
        if self.pretrained_cnn is not None:
            s_t = get_pretrained_cnn_output_for_action(s_t, self.beat_length, self.pretrained_cnn)

        action = to_numpy(
            self.actor(to_tensor(np.array([s_t])))
        ).squeeze(0)
        # add a bit of noise
        action += self.is_training * max(self.epsilon, 0) * np.random.normal(loc=self.a_noise_mean, scale=self.a_noise_std)
        # now clip the action to the valid interval and convert it to integer
        # action = np.clip(int(action), 0, self.action_upper_range - 1)
        action = np.array(action).astype(int)[0]
        if decay_epsilon:
            self.epsilon -= self.depsilon
        self.a_t = action
        return action

    def random_action(self):
        # not sure if this is the best idea
        action =  np.random.randint(0, self.action_upper_range)
        self.a_t = action
        return action

    def reset(self, obs):
        self.s_t = obs

    def load_weights(self, output):
        if output is None:
            return
        if self.pretrained_cnn is not None:
            self.pretrained_cnn.load_state_dict(
                torch.load('{}/finetuned_cnn.pkl'.format(output))
            )
        self.actor.load_state_dict(
            torch.load('{}/actor.pkl'.format(output))
        )

        self.critic.load_state_dict(
            torch.load('{}/critic.pkl'.format(output))
        )

    def save_model(self, output):
        if self.pretrained_cnn is not None:
            torch.save(self.pretrained_cnn.state_dict(),
                       '{}/finetuned_cnn.pkl'.format(output))

        torch.save(self.actor.state_dict(),
                   '{}/actor.pkl'.format(output))
        torch.save(
            self.critic.state_dict(),
            '{}/critic.pkl'.format(output))

    def cuda(self):
        if self.pretrained_cnn is not None:
            self.pretrained_cnn.cuda()
            self.pretrained_cnn_target.cuda()
        self.actor.cuda()
        self.actor_target.cuda()
        self.critic.cuda()
        self.critic_target.cuda()

    def seed(self, s):
        torch.manual_seed(s)
        if self.USE_CUDA:
            torch.cuda.manual_seed(s)












