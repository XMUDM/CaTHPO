import random
import torch
import torch.optim
import time
import numpy as np
import os
from policies import NeuralAF
from batch_recorder import BatchRecorder, Transition

import logging

logging.basicConfig(level=logging.DEBUG,
                    filename='train.log',
                    filemode='a',
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')


class PPO:
    def __init__(self, env, config):
        self.config = config
        # set up the environment (only for reading out observation and action spaces)
        self.env = env
        self.env.do_local_af_opt = config.do_local_af_opt
        self.set_all_seeds()

        # policies, optimizer
        self.device = torch.device("cpu")
        self.pi = NeuralAF(self.env, self.env.feat2dim, deterministic=False, safe=config.do_safe_control).to(self.device)
        self.old_pi = NeuralAF(self.env, self.env.feat2dim, deterministic=False, safe=config.do_safe_control).to(self.device)
        self.optimizer = torch.optim.Adam(self.pi.parameters(), lr=0.0001)
        # recode training info
        self.trn_info = TrainingInfo()

        self.t_batch = None

        self.batch_recorder = BatchRecorder(self)

    def set_all_seeds(self):
        np.random.seed(self.config.seed)
        random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed_all(self.config.seed)

    def optimize_on_batch(self):
        now = time.time()

        self.old_pi.load_state_dict(self.pi.state_dict())

        for ep in range(self.config.epochs):
            loss_ppo_ep = 0
            loss_value_ep = 0
            loss_ent_ep = 0
            loss_ep = 0

            n_batches = 0
            for batch in self.batch_recorder.iterate(self.config.batch_size):
                transitions = Transition(*zip(*batch))
                states = list(zip(*transitions.state))
                states[0] = np.array(states[0]).astype(np.float32).squeeze(1)
                states[1] = [states[1][i][0] for i in range(len(states[1]))] if len(states[1][0]) > 0 else states[1]
                states[2] = [states[2][i][0] for i in range(len(states[2]))] if len(states[2][0]) > 0 else states[2]
                actions = torch.from_numpy(np.stack(transitions.action).astype(np.float32)).to(self.device)
                tdlamrets = torch.from_numpy(np.array(transitions.tdlamret).astype(np.float32)).to(self.device)
                advs = torch.from_numpy(np.array(transitions.adv).astype(np.float32)).to(self.device)

                # normalize advantages
                advs_std = torch.std(advs, unbiased=False)
                if not advs_std == 0 and not torch.isnan(advs_std):
                    advs = (advs - torch.mean(advs)) / advs_std

                # compute values and entropies at current theta, and logprobs at current and old theta
                with torch.no_grad():
                    _, logprobs_old, _ = self.old_pi.predict_vals_logps_ents(states=states, actions=actions)
                vpreds, logprobs, entropies = self.pi.predict_vals_logps_ents(states=states, actions=actions)
                assert logprobs_old.dim() == vpreds.dim() == logprobs.dim() == entropies.dim() == 1

                # ppo-loss
                ratios = torch.exp(logprobs - logprobs_old)
                clipped_ratios = ratios.clamp(1 - self.config.epsilon, 1 + self.config.epsilon)
                advs = advs.squeeze()
                loss_cpi = ratios * advs
                assert loss_cpi.dim() == 1
                loss_clipped = clipped_ratios * advs
                assert loss_clipped.dim() == 1
                loss_ppo = -torch.mean(torch.min(loss_cpi, loss_clipped))

                # value-function loss
                loss_value = torch.mean((vpreds - tdlamrets) ** 2)

                # entropy loss
                loss_ent = -torch.mean(entropies)

                loss = loss_ppo + self.config.value_coeff * loss_value + self.config.value_coeff * loss_ent

                with torch.no_grad():
                    loss_ppo_ep += loss_ppo
                    loss_value_ep += loss_value
                    loss_ent_ep += loss_ent
                    loss_ep += loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.trn_info.n_optsteps += 1

                n_batches += 1

            loss_ppo_avg = loss_ppo_ep / n_batches
            loss_value_avg = loss_value_ep / n_batches
            loss_ent_avg = loss_ent_ep / n_batches
            loss_avg = loss_ep / n_batches
            iter_log_str = "   loss_ppo = {: .4g}, loss_value = {: .4g}, loss_ent = {: .4g}, loss = {: .4g}\n".format(
                loss_ppo_avg, loss_value_avg, loss_ent_avg, loss_avg)
            logging.info(iter_log_str)
        t_optim = time.time() - now
        iter_log_str = "  Took {:.2f}s".format(t_optim)
        logging.info(iter_log_str)

        return t_optim

    def train(self):
        os.makedirs("model_save", exist_ok=True)
        while self.trn_info.n_timesteps < self.config.max_steps:

            # store transitions
            record_time = self.batch_recorder.record_batch()

            # process information
            self.trn_info.t_train += record_time
            batch_info = self.batch_recorder.get_batch_info()
            self.trn_info.n_timesteps += len(self.batch_recorder)
            self.trn_info.batch_info = batch_info.__dict__
            self.trn_info.avg_step_rews = np.append(self.trn_info.avg_step_rews, batch_info.avg_step_reward)
            self.trn_info.avg_init_rews = np.append(self.trn_info.avg_init_rews, batch_info.avg_initial_reward)
            self.trn_info.avg_term_rews = np.append(self.trn_info.avg_term_rews, batch_info.avg_terminal_reward)
            self.trn_info.avg_ep_rews = np.append(self.trn_info.avg_ep_rews, batch_info.avg_ep_reward)
            self.trn_info.progress = 100 * self.trn_info.n_timesteps / self.config.max_steps
            batch_info.time = record_time

            # self.store_model()
            logging.info(self.trn_info.__dict__)

            t_optim = self.optimize_on_batch()
            self.store_model()

            self.trn_info.t_train += t_optim

            self.trn_info.n_iters += 1

    def store_model(self):
        with open(os.path.join("model_save/policy" + ".pth", "wb")) as f:
            torch.save(self.pi.state_dict(), f)
        print("============save model policy_", str(self.trn_info.n_iters), " success================")


class TrainingInfo:
    n_timesteps = 0
    n_optsteps = 0
    n_iters = 0
    t_train = 0
    batch_info = None
    avg_step_rews = np.array([])
    avg_init_rews = np.array([])
    avg_term_rews = np.array([])
    avg_ep_rews = np.array([])
    progress = 0
