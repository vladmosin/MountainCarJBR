import gym
import matplotlib.pyplot as plt

env = gym.make('MountainCarContinuous-v0')

first_period = 10
second_period = 34

first_period_action = 0.8
second_period_action = -0.99
last_period_action = 0.9

target_update = 10


def select_action(curr_time):
    if curr_time < first_period:
        return first_period_action
    elif curr_time < second_period:
        return second_period_action
    else:
        return last_period_action


def play(max_steps, render=False):
    result_rewards = []

    for i in range(max_steps):
        if i % target_update == 0:
            total_reward = play_step(render)
        else:
            total_reward = play_step(False)
        result_rewards.append(total_reward)

    return result_rewards


def play_step(render=False):
    env.reset()
    total_reward = 0
    curr_time = 0
    done = False
    while not done:
        if render:
            env.render()

        action = select_action(curr_time)
        state, reward, done, _ = env.step([action])
        total_reward += reward

        curr_time += 1
    return total_reward


max_steps = 1000
result_reward = play(max_steps, False)
print(sum(result_reward) / len(result_reward))
plt.plot(list(range(len(result_reward))), result_reward)
plt.show()

env.close()
