import json
import numpy as np
from util import copula_transform
import pickle
import GPy
import torch
from astnn import build_astnn
from TransformerEnc import make_transformer_encoder


def get_cos(a, b):
    dot = a * b
    a_len = np.linalg.norm(a)
    b_len = np.linalg.norm(b)
    cos = np.sum(dot) / (a_len * b_len)
    return cos

#attention
workload2attention = pickle.load(open('data/w2attention_pretrain.pkl', 'rb'))
# ast
json_str = open('data/workload2idx_tree.json').read()
workload2idx_tree = json.loads(json_str)
# seq
workload2codeidx = pickle.load(open('data/workload2codeidx.pkl', 'rb'))

class TGP():
    def __init__(
            self,
            env,
            training_data=None,
            bandwidth: float = 0.1,
            variance_mode: str = 'target',
            normalization: str = 'None',
    ):
        self.env = env
        self.D = env.D
        self.training_data = training_data
        self.bandwidth = bandwidth
        self.variance_mode = variance_mode
        self.normalization = normalization

        if self.normalization not in ['None', 'mean/var', 'Copula']:
            raise ValueError(self.normalization)

        self.base_models = []
        self.base_models_workload = []
        self.base_models_dataX = []
        self.base_models_datay = []
        self.base_models_datasize = []

        self.kernel = GPy.kern.RBF(input_dim=self.D,
                                   variance=1.,
                                   lengthscale=1.,
                                   ARD=True)

        if training_data is not None:
            base_models = []
            base_models_workload = []
            base_models_dataX = []
            base_models_datay = []
            base_models_datasize = []
            for task in training_data:
                taskworkload = training_data[task]['workload']
                taskdatasize = training_data[task]['datasize']
                base_models_workload.append(taskworkload)
                base_models_datasize.append(taskdatasize)

                Y = training_data[task]['y']
                if self.normalization == 'mean/var':
                    mean = Y.mean()
                    std = Y.std()
                    if std == 0:
                        std = 1
                    y_scaled = (Y - mean) / std
                    y_scaled = y_scaled.flatten()
                elif self.normalization == 'Copula':
                    y_scaled = copula_transform(Y)
                elif self.normalization == 'None':
                    y_scaled = Y
                else:
                    raise ValueError(self.normalization)
                configs = training_data[task]['configurations']
                X = np.array(configs)
                y_scaled = np.array(y_scaled).reshape(y_scaled.shape[0], 1)
                base_models_dataX.append(X)
                base_models_datay.append(y_scaled)
                model = GPy.models.gp_regression.GPRegression(X, y_scaled,
                                                              noise_var=1.,
                                                              kernel=self.kernel,
                                                              normalizer=False)
                base_models.append(model)
            self.base_models = base_models
            self.base_models_workload = base_models_workload
            self.base_models_dataX = base_models_dataX
            self.base_models_datay = base_models_datay
            self.base_models_datasize = base_models_datasize

        self.weights_over_time = []
        self.target_model = None
        self.target_model_workload = None
        self.target_model_dataX = None
        self.target_model_datay = None
        self.target_model_datasize = None

    def get_safe_threshold(self):
        weights = np.zeros(len(self.model_list_))
        weights[-1] = workload2attention[self.target_model_workload][self.target_model_workload]
        for model_idx, model in enumerate(self.base_models):
            weights[model_idx] = workload2attention[self.target_model_workload][self.base_models_workload[model_idx]]
        weights /= np.sum(weights)

        h = weights[-1] * np.percentile(self.target_model_datay, 25)
        for model_idx, model in enumerate(self.base_models):
            h += weights[model_idx] * np.percentile(self.base_models_datay[model_idx], 25)
        return h

    def get_astembedding(self, workload):
        inp = [workload2idx_tree[workload] for _ in range(2)]
        return torch.flatten(self.env.pi.astnn(inp))

    def get_seqembedding(self, workload):
        seq = [workload2codeidx[workload] for _ in range(2)]
        return torch.flatten(self.env.pi.seqEncoder(seq))

    def get_code_similarity(self, workload_1, workload_2):
        astfeat_1 = self.get_astembedding(workload_1).detach().numpy()
        seqfeat_1 = self.get_seqembedding(workload_1).detach().numpy()
        feat_1 = np.concatenate((astfeat_1, seqfeat_1), axis=0)

        astfeat_2 = self.get_astembedding(workload_2).detach().numpy()
        seqfeat_2 = self.get_seqembedding(workload_2).detach().numpy()
        feat_2 = np.concatenate((astfeat_2, seqfeat_2), axis=0)

        cos = get_cos(feat_1, feat_2)
        # print('============', workload_1, workload_2, 'code cos: ', cos)
        return cos

    def train(self, X: np.ndarray, Y: np.ndarray, target_workload, target_datasize):
        if self.normalization == 'mean/var':
            Y = Y.flatten()
            mean = Y.mean()
            std = Y.std()
            if std == 0:
                std = 1

            y_scaled = (Y - mean) / std
            self.Y_std_ = std
            self.Y_mean_ = mean
        elif self.normalization in ['None', 'Copula']:
            self.Y_mean_ = 0.
            self.Y_std_ = 1.
            y_scaled = Y
            if self.normalization == 'Copula':
                y_scaled = copula_transform(Y)
        else:
            raise ValueError(self.normalization)

        target_model = GPy.models.gp_regression.GPRegression(X, y_scaled,
                                                      noise_var=1.,
                                                      kernel=self.kernel,
                                                      normalizer=False)
        self.target_model = target_model
        self.target_model_workload = target_workload
        self.target_model_dataX = X
        self.target_model_datay = y_scaled
        self.target_model_datasize = target_datasize

        self.model_list_ = self.base_models + [target_model]

        weights = np.zeros(len(self.model_list_))
        weights[-1] = 0.75 * workload2attention[self.target_model_workload][self.target_model_workload]

        discordant_pairs_per_task = {}
        for model_idx, model in enumerate(self.base_models):
            if X.shape[0] < 2:
                weights[model_idx] = 0.75 * workload2attention[self.base_models_workload[model_idx]][self.target_model_workload]
            else:
                mean, _ = model.predict_noiseless(X)
                discordant_pairs = 0
                total_pairs = 0
                for i in range(X.shape[0]):
                    for j in range(i + 1, X.shape[0]):
                        if (Y[i] < Y[j]) ^ (mean[i] < mean[j]):
                            discordant_pairs += 1
                        total_pairs += 1
                t = discordant_pairs / total_pairs / self.bandwidth
                discordant_pairs_per_task[model_idx] = discordant_pairs
                if t < 1:
                    weights[model_idx] = 0.75 * (1 - t ** 2)* workload2attention[self.base_models_workload[model_idx]][self.target_model_workload]
                else:
                    weights[model_idx] = 0

        weights /= np.sum(weights)
        self.weights_ = weights

        self.weights_over_time.append(weights)
        return self

    def predict(self, X: np.ndarray):
        weighted_means = []
        weighted_covars = []

        non_zero_weight_indices = (self.weights_ ** 2 > 0).nonzero()[0]
        non_zero_weights = self.weights_[non_zero_weight_indices]
        # re-normalize
        non_zero_weights /= non_zero_weights.sum()

        for non_zero_weight_idx in range(non_zero_weight_indices.shape[0]):
            raw_idx = non_zero_weight_indices[non_zero_weight_idx].item()
            weight = non_zero_weights[non_zero_weight_idx]
            mean, covar = self.model_list_[raw_idx].predict_noiseless(X)

            weighted_means.append(weight * mean)

            if self.variance_mode == 'average':
                weighted_covars.append(covar * weight ** 2)
            elif self.variance_mode == 'target':
                pass
            else:
                raise ValueError()


        mean_x = np.sum(np.stack(weighted_means), axis=0) * self.Y_std_ + self.Y_mean_
        if self.variance_mode == 'average':
            covar_x = np.sum(weighted_covars, axis=0) * (self.Y_std_ ** 2)
        elif self.variance_mode == 'target':
            _, covar_x = self.model_list_[-1].predict_noiseless(X)

        return mean_x, covar_x

    def reset(self):
        if self.target_model is not None:
            for model_idx, model in enumerate(self.base_models):
                if self.base_models_datasize[model_idx] == self.target_model_datasize and self.base_models_workload[model_idx] == self.target_model_workload:
                    self.base_models_dataX[model_idx] = np.concatenate((self.base_models_dataX[model_idx], self.target_model_dataX))
                    self.base_models_datay[model_idx] = np.concatenate((self.base_models_datay[model_idx], self.target_model_datay))
                    self.base_models[model_idx] = GPy.models.gp_regression.GPRegression(self.base_models_dataX[model_idx],
                                                                                        self.base_models_datay[model_idx],
                                                                                        noise_var=1.,
                                                                                        kernel=self.kernel,
                                                                                        normalizer=False)
                    self.weights_over_time = []
                    self.target_model = None
                    self.target_model_workload = None
                    self.target_model_dataX = None
                    self.target_model_datay = None
                    self.target_model_datasize = None
                    return

            self.base_models = self.base_models + [self.target_model]
            self.base_models_workload = self.base_models_workload + [self.target_model_workload]
            self.base_models_dataX = self.base_models_dataX + [self.target_model_dataX]
            self.base_models_datay = self.base_models_datay  + [self.target_model_datay]
            self.base_models_datasize = self.base_models_datasize + [self.target_model_datasize]

            self.weights_over_time = []
            self.target_model = None
            self.target_model_workload = None
            self.target_model_dataX = None
            self.target_model_datay = None
            self.target_model_datasize = None



if __name__ == '__main__':
    pass