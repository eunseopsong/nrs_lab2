# nrs_lab2/tasks/manager_based/nrs_lab2/mdp/rewards.py

def bc_zero_reward(env, scene):
    # Behavior Cloning은 보상 없이 imitation으로만 학습
    return 0.0
