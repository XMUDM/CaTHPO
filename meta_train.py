import torch
from environment import Environment
from ppo import PPO

class Config:
    seed = 2022
    epochs = 15
    buffer_size = 32
    batch_size = 32
    max_steps = buffer_size * 15
    epsilon = 0.15
    value_coeff = 1.0
    ent_coeff = 0.01
    gamma = 0.98
    lambda_ = 0.98
    do_local_af_opt = True
    do_transfer_gp = True
    metadata_path = None
    do_safe_control = True
    save_path = 'train_data_rs.csv'

def train():
    env = Environment(Config(), testing=False)
    ppo = PPO(env, Config())
    ppo.train()
    print()

if __name__ == '__main__':
    train()