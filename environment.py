import pickle
import time
import random
import json
import sobol_seq
import csv
import GPy
import gym
from gym import spaces
from run_action import run_bench
from util import *
from transfer_gp import TGP

import logging

logging.basicConfig(level=logging.DEBUG,
                    filename='train.log',
                    filemode='a',
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

workloads = ['BiVAE', 'VAECF', 'NCF', 'LightGCN', 'CML', 'CDAE', 'UAutoRec', 'IAutoRec']

datasets = [1, 2, 3, 4]
datasets2itemnum = {
    1: 4920,
    2: 927,
    3: 2624,
    4: 6901,
    5: 4582
}
datasets2usernum = {
    1: 478,
    2: 2925,
    3: 1928,
    4: 1956,
    5: 3855
}
datasets2ratingnum = {
    1: 62452,
    2: 127917,
    3: 181567,
    4: 256017,
    5: 448712
}

#Pretrain
workload2pretrain = pickle.load(open('data/w2pretrainfeat.pkl', 'rb'))

# AST (for ablation study)
json_str = open('data/workload2idx_tree.json').read()
workload2ast = json.loads(json_str)

#SEQ (for ablation study)
workload2codeidx = pickle.load(open('data/workload2codeidx.pkl', 'rb'))

#codeBERT (for ablation study)
workload2codebert = pickle.load(open('data/workload2codebertfeat.pkl', 'rb'))

a_low = np.array([5, 0, 64])
a_high = np.array([512, 1000, 1024])

last_log = ""

hyper_names = ['ld', 'lr', 'bs']

class Environment(gym.Env):

    def __init__(self, config, testing=False):
        # testing=True just for test the environment
        self.config = config
        self.testing = testing
        self.save_path = config.save_path
        self.do_tgp = config.do_transfer_gp
        self.a_low = a_low
        self.a_high = a_high
        # action dim
        self.D = a_high.shape[0]
        self.action_space = spaces.Box(low=self.a_low, high=self.a_high, dtype=np.int)
        self.state = None
        self.default_action = self.a_low
        self.workload = None
        self.datasize = None
        self.feat2dim = {"data_feat": 3,
                         # "code_bert": 64,
                         "pretrain": 128,
                         "mean": 1,
                         "std": 1,
                         "timestep": 1,
                         "budget": 1,
                         "x": self.D,
                         # "code_ast": 64,
                         # "code_seq": 64,
                         }

        self.feature_dim = sum(self.feat2dim.values())
        idx = 0
        for (k, v) in self.feat2dim.items():
            if k == "timestep":
                self.t_idx = idx
            if k == "budget":
                self.T_idx = idx
            idx += v

        self.af = None
        self.pi = None

        # optimization step
        self.t = None
        # the training data
        self.X = self.Y = None  # None means empty
        self.gp_is_empty = True
        # the surrogate GP
        self.kernel = None
        self.gp = None
        self.tgp_model = None

        self.do_local_af_opt = True
        if self.do_local_af_opt:
            self.domain = np.zeros((self.D,))
            self.domain = np.stack([self.domain, np.ones(self.D, )], axis=1)
            self.xi_t = None
            self.af_opt_startpoints_t = None
            self.af_maxima_t = None
            N_MS_per_dim = 2
            # self.multistart_grid, _ = create_uniform_grid(self.domain, N_MS_per_dim)
            self.multistart_grid = create_random_x(self.D, 32768)
            self.N_MS = self.multistart_grid.shape[0]
            self.k = 5
            self.cardinality_xi_local_t = self.k
            self.cardinality_xi_global_t = self.N_MS
            self.cardinality_xi_t = self.cardinality_xi_local_t + self.cardinality_xi_global_t

            self.N_LS = 1000
            self.local_search_grid = sobol_seq.i4_sobol_generate(self.D, self.N_LS)
            self.af_max_search_diam = 2 * 1 / N_MS_per_dim

        else:
            self.cardinality_xi_t = 256
            self.xi_t = sobol_seq.i4_sobol_generate(self.D, self.cardinality_xi_t)

        # optimization step
        self.T = 16
        self.t = 0

        self.observation_space = spaces.Box(low=0.0, high=100.0,
                                            shape=(self.cardinality_xi_t, self.feature_dim),
                                            dtype=np.float32)

        if self.do_tgp:
            if config.metadata_path:
                meta_data = get_meta_data(config.metadata_path)
            else:
                meta_data = None
            self.tgp_model = TGP(self, training_data=meta_data)

        with open(self.save_path, 'a', encoding='utf8') as f:
            writer = csv.writer(f)
            writer.writerow(['Workload', 'NDCG', 'dataset'] + hyper_names + ['ld_', 'lr_', 'bs_'])

    def set_af_functions(self, af_fun):
        self.af = af_fun

    def set_pi_functions(self, pi_fun):
        self.pi = pi_fun

    def update_TGP(self):
        assert self.Y is not None
        self.tgp_model.train(self.X, self.Y, self.workload, self.datasize)

    def step(self, action_idx):
        if not isinstance(action_idx, np.ndarray):
            action_idx = np.array([action_idx])
        action = self.xi_t[action_idx, :].reshape(action_idx.size, self.D)
        new_action = self.transform_action(action)
        print('action: ', new_action)

        if not self.testing:
            code, msg, y = run_bench(self.workload, self.datasize, new_action)
        else:
            code, msg, y = 0, "testing", 0
            time.sleep(1)

        if code == 0:
            with open(self.save_path, 'a', encoding='utf8') as f:
                writer = csv.writer(f)
                writer.writerow([self.workload, y, self.datasize] + action.tolist()[0] + new_action)

            self.add_data(action, y)
            reward = self.get_reward()
            self.update_gp()
            if self.do_tgp:
                self.update_TGP()
            self.optimize_AF()
            self.get_state(self.xi_t)
        else:
            reward = 0

        done = self.t == self.T
        logging.info("reward:" + str(reward) + "\n")
        return self.state, reward, done, {}

    def transform_action(self, action):
        action = np.squeeze(action, 0)
        new_action = []
        for i in range(len(action)):
            new_action.append(int(a_low[i] + float(action[i]) * (a_high[i] - a_low[i])))
            if new_action[i] > a_high[i]:
                new_action[i] = a_high[i]
            if new_action[i] < a_low[i]:
                new_action[i] = a_low[i]
        assert self.action_space.contains(new_action), "%r (%s) invalid" % (new_action, type(new_action))
        return new_action

    def get_reward(self):
        y_diffs = self.Y
        simple_regret = np.max(y_diffs)
        reward = np.asscalar(simple_regret)
        return reward

    def add_data(self, x, y):
        x, y = np.array(x), np.array([y])
        if len(x.shape) == 1:
            x = np.expand_dims(x, 0)
        if len(y.shape) == 1:
            y = np.expand_dims(y, 0)
        if self.X is None:
            self.X = x
            self.Y = y
        else:
            self.X = np.concatenate((self.X, x), axis=0)
            self.Y = np.concatenate((self.Y, y), axis=0)
        if len(x.shape) == 2:
            self.t += x.shape[0]
        else:
            self.t += 1

    def update_gp(self):
        assert self.Y is not None
        self.gp.set_XY(self.X, self.Y)
        self.gp_is_empty = False

    def eval_gp(self, X_star):
        if len(X_star.shape) == 1:
            X_star = np.expand_dims(X_star, 0)
        # evaluate the GP on X_star
        assert X_star.shape[1] == self.D
        gp_mean, gp_var = self.gp.predict_noiseless(X_star)
        gp_mean = gp_mean[:, 0]
        gp_var = gp_var[:, 0]
        gp_std = np.sqrt(gp_var)
        return gp_mean, gp_std

    def get_state(self, X):
        if len(X.shape) == 1:
            X = np.expand_dims(X, 0)
        idx = 0
        vec_state = np.zeros((X.shape[0], self.feature_dim), dtype=np.float32)

        if self.do_tgp:
            gp_mean, gp_std = self.tgp_model.predict(X)
        else:
            gp_mean, gp_std = self.eval_gp(X)

        for (feat, dim) in self.feat2dim.items():
            if feat == "code_bert":
                codebert = workload2codebert[self.workload]
                codebert = np.array([codebert for _ in range(X.shape[0])])
                vec_state[:, idx:idx + dim] = codebert.reshape(X.shape[0], dim)
                idx += dim
            if feat == "pretrain":
                pretrainfeat = workload2pretrain[self.workload]
                pretrainfeat = np.array([pretrainfeat for _ in range(X.shape[0])])
                vec_state[:, idx:idx + dim] = pretrainfeat.reshape(X.shape[0], dim)
                idx += dim
            if feat == "mean":
                vec_state[:, idx:idx + 1] = gp_mean.reshape(X.shape[0], 1)
                idx += 1
            if feat == "std":
                vec_state[:, idx:idx + 1] = gp_std.reshape(X.shape[0], 1)
                idx += 1
            if feat == "timestep":
                t_vec = np.ones((X.shape[0],)) * self.t
                vec_state[:, idx] = t_vec
                idx += 1
            if feat == "budget":
                T_vec = np.ones((X.shape[0],)) * self.T
                vec_state[:, idx] = T_vec
                idx += 1
            if feat == "data_feat":
                datafeat = [[datasets2itemnum[self.datasize], datasets2usernum[self.datasize], datasets2ratingnum[self.datasize]]]
                datafeat = np.array([datafeat for _ in range(X.shape[0])])
                vec_state[:, idx:idx + dim] = datafeat.reshape(X.shape[0], dim)
                idx += dim
            self.state = [vec_state]
            if "code_ast" in self.feat2dim.keys():
                code_ast = [workload2ast[self.workload]]
                self.state.append(code_ast)
            else:
                self.state.append([])
            if "code_seq" in self.feat2dim.keys():
                code_seq = [workload2codeidx[self.workload]]
                self.state.append(code_seq)
            else:
                self.state.append([])
        return self.state

    def get_incumbent(self):
        if self.Y is None:
            Y = np.array([0])
        else:
            Y = self.Y

        incumbent = np.max(Y)
        return incumbent

    def reset(self, workload=None, datasize=None):
        self.X = self.Y = None
        # reset workload and data size
        if workload is None:
            self.workload = random.choice(workloads)
        else:
            self.workload = workload
        logging.info("============" + self.workload + "============")
        if datasize is None:
            self.datasize = random.choice(datasets)
        else:
            self.datasize = datasize

        # reset step counter
        self.t = 0
        # reset gp
        X = np.zeros((1, self.D))
        Y = np.zeros((1, 1))
        self.kernel = GPy.kern.RBF(input_dim=self.D,
                                   variance=1.,
                                   lengthscale=1.,
                                   ARD=True)
        self.gp = GPy.models.gp_regression.GPRegression(X, Y,
                                                        noise_var=1.,
                                                        kernel=self.kernel,
                                                        normalizer=False)
        if self.do_tgp:
            self.tgp_model.reset()
            self.tgp_model.train(X, Y, self.workload, self.datasize)

        self.optimize_AF()
        # get state
        self.get_state(self.xi_t)
        return self.state

    def optimize_AF(self):
        if self.do_local_af_opt:
            # obtain maxima of af
            self.get_af_maxima()
            self.xi_t = np.concatenate((self.af_maxima_t, self.multistart_grid), axis=0)
            assert self.xi_t.shape[0] == self.cardinality_xi_t
        else:
            pass

    def get_af_maxima(self):
        state_at_multistarts = self.get_state(self.multistart_grid)
        af_at_multistarts = self.af(state_at_multistarts)
        self.af_opt_startpoints_t = self.multistart_grid[np.argsort(-af_at_multistarts)[:self.k, ...]]

        local_grids = [scale_from_unit_square_to_domain(self.local_search_grid,
                                                        domain=get_cube_around(x,
                                                                               diam=self.af_max_search_diam,
                                                                               domain=self.domain))
                       for x in self.af_opt_startpoints_t]
        local_grids = np.concatenate(local_grids, axis=0)
        state_on_local_grid = self.get_state(local_grids)
        af_on_local_grid = self.af(state_on_local_grid)
        self.af_maxima_t = local_grids[np.argsort(-af_on_local_grid)[:self.cardinality_xi_local_t]]

        assert self.af_maxima_t.shape[0] == self.cardinality_xi_local_t

    def render(self, mode='human'):
        return None

    def close(self):
        return None




