import torch
from environment import Environment
from ppo import PPO
from environment import workloads

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
    metadata_path = 'train_data_rs.csv'
    do_safe_control = True
    save_path = 'test_data_rs.csv'

def evl():
    env = Environment(testing=False)
    ppo = PPO(env, Config())
    ppo.pi.load_state_dict(torch.load("model_save/policy.pth"))
    ppo.batch_recorder.size = len(workloads) * env.T
    for w in workloads:
        ppo.batch_recorder.eval_one(w, 5)
    ppo.batch_recorder.add_tdlamret_and_adv()
    batch_info = ppo.batch_recorder.get_batch_info().__dict__
    print(batch_info)

if __name__ == '__main__':
    evl()