import copy
import random

import gym
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from torch import optim

env = gym.make("MountainCar-v0")

batch_size = 256
neuron_number = 40
gamma = 0.998
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_weights(layer):
    if type(layer) == nn.Linear:
        nn.init.xavier_normal_(layer.weight)


def create_model():
    return nn.Sequential(
        nn.Linear(2, neuron_number),
        nn.ReLU(),
        nn.Linear(neuron_number, neuron_number),
        nn.ReLU(),
        nn.Linear(neuron_number, 3)
    )


def prepare_model(model):
    model.train()
    model.to(device)


def init_model():
    model = create_model()
    target_model = copy.deepcopy(model)
    model.apply(init_weights)

    prepare_model(model)
    prepare_model(target_model)

    optimizer = optim.Adam(model.parameters(), lr=0.00003)

    return model, target_model, optimizer


def select_action(state, model, epsilon):
    if random.random() < epsilon:
        return random.randint(0, 2)
    return model(torch.tensor(state).to(device).float().unsqueeze(0))[0].max(0)[1].view(1, 1).item()


def fit(batch, model, target_model, optimizer):
    state, action, reward, next_state, done = batch
    state = torch.tensor(state).to(device).float()
    next_state = torch.tensor(next_state).to(device).float()
    reward = torch.tensor(reward).to(device).float()
    action = torch.tensor(action).to(device)

    target_q = torch.zeros(reward.size()[0]).float().to(device)
    with torch.no_grad():
        target_q[done] = target_model(next_state).max(1)[0].detach()
    target_q = reward + target_q * gamma

    q = model(state).gather(1, action.unsqueeze(1))

    loss = F.smooth_l1_loss(q, target_q.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()

    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


class Memory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, element):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = element
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


memory = Memory(10000)
update_target = 1000


def play(max_step):
    reward_by_percentage = []
    state = env.reset()
    model, target_model, optimizer = init_model()
    for step in range(1, max_step):
        epsilon = 1 - 0.9 * step / max_step
        action = select_action(state, model, epsilon)
        new_state, reward, done, _ = env.step(action)

        if done:
            memory.push((state, action, reward, state, done))
            state = env.reset()
        else:
            memory.push((state, action, reward + abs(new_state[1]), new_state, done))
            state = new_state

        if step > batch_size:
            fit(list(zip(*memory.sample(batch_size))), model, target_model, optimizer)

        if step % update_target == 0:
            target_model = copy.deepcopy(model)
            print("Progress", step // update_target, "of", max_step // update_target)
            state = env.reset()
            total_reward = 0
            done = False
            while not done:
                epsilon = 0.0
                if (step // update_target) % 2 == 0:
                    env.render()
                action = select_action(state, model, epsilon)
                state, reward, done, _ = env.step(action)
                total_reward += reward
            reward_by_percentage.append(total_reward)
    return reward_by_percentage


result_reward = play(100000)
plt.plot(list(range(len(result_reward))), result_reward)
plt.show()

env.close()
