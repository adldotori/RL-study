# 필수:
# 왼쪽 미로에서 dyna q learning 을 이용해서 goal에 도착하는데 필요한 step수 그래프를 그리기.
# x축 episode y축 step.
# planning step을 0(naive q learning),5,50으로 오른쪽과 같은 그래프를 그리세요.


import copy
import random

import matplotlib.pyplot as plt
import numpy as np

# 0: left, 1: down, 2: right, 3: up
actions = [0, 1, 2, 3]

map = [
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1, 0, 0, 0, 0, 0, 0, 0, 0, 1000, -1],
    [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
    [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
    [-1, 0, -1, -1, -1, -1, -1, -1, -1, 0, -1],
    [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
    [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
]


def get_next_state(state, action):
    # state: [y, x]
    # action: 0, 1, 2, 3
    next_state = copy.deepcopy(state)
    if action == 0:
        next_state[1] -= 1
    elif action == 1:
        next_state[0] += 1
    elif action == 2:
        next_state[1] += 1
    elif action == 3:
        next_state[0] -= 1

    if (
        next_state[0] < 0
        or next_state[0] >= len(map)
        or next_state[1] < 0
        or next_state[1] >= len(map[0])
    ):
        return state
    if map[next_state[0]][next_state[1]] == -1:
        return state
    return next_state


def is_available(state, action):
    next_state = get_next_state(state, action)
    if next_state == state:
        return False
    return True


def get_reward(state, action):
    return map[state[0]][state[1]]


def get_action(state, q_table, epsilon):
    available_actions = []
    for action in actions:
        if is_available(state, action):
            available_actions.append(action)
    if len(available_actions) == 0:
        return None
    if random.random() < epsilon:
        return random.choice(available_actions)
    else:
        return np.random.choice(
            np.where(q_table[state[0]][state[1]] == q_table[state[0]][state[1]].max())[
                0
            ]
        )


def dyna_q_learning(n, alpha, epsilon, gamma, planning_step):
    q_table = np.zeros((len(map), len(map[0]), len(actions)))
    model = {}
    steps = []
    for episode in range(100):
        state = [6, 4]
        step = 0
        while True:
            action = get_action(state, q_table, epsilon)
            if action is None:
                break
            next_state = get_next_state(state, action)
            reward = get_reward(next_state, action)

            q_table[state[0]][state[1]][action] += alpha * (
                reward
                + gamma * np.max(q_table[next_state[0]][next_state[1]])
                - q_table[state[0]][state[1]][action]
            )
            model[(state[0], state[1], action)] = (reward, next_state[0], next_state[1])

            print(
                "episode: {} step: {} state: {} action: {} next_state: {} reward: {} q_table: {}".format(
                    episode,
                    step,
                    state,
                    action,
                    next_state,
                    reward,
                    q_table[state[0]][state[1]],
                ),
            )

            if reward > 0:
                break

            for _ in range(planning_step):
                state = random.choice(list(model.keys()))
                reward, next_y, next_x = model[state]
                q_table[state[0]][state[1]][state[2]] += alpha * (
                    reward
                    + gamma * np.max(q_table[next_y][next_x])
                    - q_table[state[0]][state[1]][state[2]]
                )
            state = next_state
            step += 1
        steps.append(step)
    return steps


def main():
    n = 10
    alpha = 0.1
    epsilon = 0.1
    gamma = 0.95
    planning_steps = [0, 5, 50]

    for planning_step in planning_steps:
        steps = dyna_q_learning(n, alpha, epsilon, gamma, planning_step)
        print("planning step: {}".format(planning_step))
        plt.plot(steps, label="planning step: {}".format(planning_step))
    plt.legend()
    plt.title("Dyna-Q Learning")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.show()
    # save image
    plt.savefig("images/6_DynaQ.png")


if __name__ == "__main__":
    main()
