# Copyright (c) 2019 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# ******************************************************************
# util.py
# Utilities for the MetaBO framework.
# ******************************************************************
import numpy as np
import scipy as sp
from scipy.stats import norm


def get_meta_data(path):
    # Sample data for each base task
    import numpy as np
    import pandas as pd
    df = pd.read_csv(path, sep=',',  low_memory=False)
    df.sort_values(by=['Workload', 'dataset'])
    group = df.groupby(['Workload', 'dataset'])
    task_size = [int(i) for i in group.size()]
    data_by_task = {}
    start = 0
    print(df.loc(start))

    for task in range(len(task_size)):
        taskworkload = df.loc[start]['Workload']
        taskdatasize = df.loc[start]['dataset']

        train_y = []
        configurations = []
        for j in range(start, start+task_size[task]):
            train_y.append(float(df.loc[j]['NDCG']))
            conf = []
            conf.append(float(df.loc[j]['ld']))
            conf.append(float(df.loc[j]['lr']))
            conf.append(float(df.loc[j]['bs']))
            configurations.append(conf)
        train_y = np.array(train_y)
        data_by_task[task] = {
            'workload': taskworkload,
            'datasize': taskdatasize,
            'configurations': configurations,
            'y': train_y,
        }
        start = start + task_size[task]
    return data_by_task


def copula_transform(values: np.ndarray) -> np.ndarray:

    """Copula transformation from "A Quantile-based Approach for Hyperparameter Transfer Learning"
    by  Salinas, Shen and Perrone, ICML 2020"""

    quants = (sp.stats.rankdata(values.flatten()) - 1) / (len(values) - 1)
    cutoff = 1 / (4 * np.power(len(values), 0.25) * np.sqrt(np.pi * np.log(len(values))))
    quants = np.clip(quants, a_min=cutoff, a_max=1-cutoff)
    # Inverse Gaussian CDF
    rval = np.array([sp.stats.norm.ppf(q) for q in quants]).reshape((-1, 1))
    return rval

def create_uniform_grid(domain, N_samples_dim):
    D = domain.shape[0]
    x_grid = []
    for i in range(D):
        x_grid.append(np.linspace(domain[i, 0], domain[i, 1], N_samples_dim))
    X_mesh = np.meshgrid(*x_grid)
    X = np.vstack(X_mesh).reshape((D, -1)).T

    return X, X_mesh


def create_random_x(D, samples):
    return np.random.rand(samples, D)


def scale_from_unit_square_to_domain(X, domain):
    # X contains elements in unit square, stretch and translate them to lie domain
    return X * domain.ptp(axis=1) + domain[:, 0]


def scale_from_domain_to_unit_square(X, domain):
    # X contains elements in domain, translate and stretch them to lie in unit square
    return (X - domain[:, 0]) / domain.ptp(axis=1)


def get_cube_around(X, diam, domain):
    assert X.ndim == 1
    assert domain.ndim == 2
    cube = np.zeros(domain.shape)
    cube[:, 0] = np.max((X - 0.5 * diam, domain[:, 0]), axis=0)
    cube[:, 1] = np.min((X + 0.5 * diam, domain[:, 1]), axis=0)
    return cube


if __name__ == '__main__':
    # D = 15
    # N_MS = 300000
    # x = create_random_x(D, 32768)
    # N_MS_per_dim = np.int(np.floor(N_MS ** (1 / D)))
    # domain = np.stack([np.zeros((15,)), np.ones(15, )], axis=1)
    # x, x_mesh = create_uniform_grid(domain, 2)
    #
    # res = scale_from_domain_to_unit_square(np.array([1, 2]), domain)
    #
    # print()
    print(get_meta_data('test_metabo_0702.csv'))
