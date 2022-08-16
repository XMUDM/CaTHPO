import time
import torch
import random
from namedlist import namedlist
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG,
                    filename='train.log',
                    filemode='a',
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

Transition = namedlist("Transition", ["state", "action", "reward", "value", "new", "tdlamret", "adv"])


class BatchRecorder:

    def __init__(self, agent):
        self.env = agent.env
        self.pi = agent.pi
        agent.env.unwrapped.set_af_functions(af_fun=self.pi.af)
        agent.env.unwrapped.set_pi_functions(pi_fun=self.pi)
        self.memory = []
        self.next_state = None
        self.next_value = None
        self.next_new = None
        self.gamma = agent.config.gamma
        self.lam = agent.config.lambda_
        self.initial_rewards = []
        self.terminal_rewards = []
        self.size = agent.config.buffer_size
        self.reward_sum = 0
        self.n_new = 0

    def clear(self):
        self.memory = []
        self.reward_sum = 0
        self.n_new = 0
        self.initial_rewards = []
        self.terminal_rewards = []
        self.next_new = None
        self.next_state = None
        self.next_value = None

    def push(self, state, action, reward, value, new):
        assert not self.is_full()
        self.memory.append(Transition(state, action, reward, value, new, None, None))
        self.reward_sum += reward
        self.n_new += int(new)

    def is_full(self):
        return len(self) == self.size

    def is_empty(self):
        return len(self) == 0

    def __len__(self):
        return len(self.memory)

    def record_batch(self):
        start = time.time()
        if self.next_state is None:
            state = self.env.reset()
            new = 1
        else:
            state = self.next_state
            new = self.next_new
        self.clear()
        while not self.is_full():
            with torch.no_grad():
                action, value = self.pi.act(state)
            action, value = action.numpy(), value.numpy()
            next_state, reward, done, _ = self.env.step(action)
            self.push(state, action, reward, value, new)
            if done:
                state = self.env.reset()
                new = 1
            else:
                state = next_state
                new = 0
        self.next_new = new
        self.next_state = state
        self.next_value = value
        self.add_tdlamret_and_adv()
        return time.time() - start

    def eval_one(self, workload, rows):
        state = self.env.reset(workload, rows)
        new = 1
        while not self.is_full():
            with torch.no_grad():
                action, value = self.pi.act(state)
            action, value = action.numpy(), value.numpy()
            next_state, reward, done, _ = self.env.step(action)
            self.push(state, action, reward, value, new)
            if done:
                break
            else:
                state = next_state
                new = 0
        self.next_new = new
        self.next_state = state
        self.next_value = value

    def add_tdlamret_and_adv(self):
        gamma, lam = self.gamma, self.lam
        assert self.is_full()
        self.initial_rewards = []  # extraction of initial rewards can happen here w/o overhead
        self.terminal_rewards = []  # extraction of terminal rewards can happen here w/o overhead
        next_new = self.next_new
        next_value = self.next_value
        next_adv = 0
        for i in reversed(range(len(self))):
            nonterminal = 1 - next_new
            value = self.memory[i].value
            reward = self.memory[i].reward
            if self.memory[i].new:
                self.initial_rewards.append(reward)
            if not nonterminal:
                self.terminal_rewards.append(reward)

            delta = -value + reward + gamma * nonterminal * next_value
            self.memory[i].adv = next_adv = delta + lam * gamma * nonterminal * next_adv
            self.memory[i].tdlamret = self.memory[i].adv + value
            next_new = self.memory[i].new
            next_value = value

    def iterate(self, minibatch_size, shuffle=True):
        assert self.is_full()
        pos = 0
        idx = list(range(len(self)))
        if shuffle:
            # we use the random state of the main process here, NO re-seeding
            random.shuffle(idx)
        while pos < len(self):
            if pos + 2 * minibatch_size > len(self):
                # enlarge the last minibatch s.t. all minibatches are at least of size minibatch_size
                cur_minibatch_size = len(self) - pos
            else:
                cur_minibatch_size = minibatch_size
            cur_idx = idx[pos:pos + cur_minibatch_size]
            yield [self.memory[i] for i in cur_idx]
            pos += cur_minibatch_size

    def get_batch_info(self):
        #assert self.is_full()
        return BatchInfo(size=len(self),
                         avg_step_reward=self.reward_sum / len(self),
                         avg_initial_reward=np.mean(self.initial_rewards),
                         avg_terminal_reward=np.mean(self.terminal_rewards),
                         avg_ep_reward=self.reward_sum / self.n_new,
                         avg_ep_len=len(self) / self.n_new,
                         n_new=self.n_new)


class BatchInfo:
    def __init__(self, size, avg_step_reward, avg_initial_reward, avg_terminal_reward, avg_ep_reward,
                 avg_ep_len, n_new):
        self.size, self.avg_step_reward, self.avg_initial_reward, self.avg_terminal_reward, self.avg_ep_reward, self.avg_ep_len, self.n_new = \
            size, avg_step_reward, avg_initial_reward, avg_terminal_reward, avg_ep_reward, avg_ep_len, n_new



