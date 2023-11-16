import math
import random

import pandas as pd


class Arm:
    def __init__(self, p: float):
        self.p = p

    def pull(self):
        return 1 if random.random() < self.p else 0


class Policy:
    def __init__(self, arms: list[Arm]):
        self.arms = arms

    def select_arm(self) -> int:
        raise NotImplementedError

    def update(self, arm: int, reward: float) -> None:
        raise NotImplementedError


class Greedy(Policy):
    def __init__(self, arms: list[Arm]):
        super().__init__(arms)
        self.N = [0] * len(arms)
        self.Q = [0.0] * len(arms)

    def select_arm(self) -> int:
        return self.Q.index(max(self.Q))

    def update(self, arm: int, reward: float) -> None:
        self.N[arm] += 1
        self.Q[arm] += (1 / self.N[arm]) * (reward - self.Q[arm])


class EpsilonGreedy(Policy):
    def __init__(self, arms: list[Arm], epsilon: float):
        super().__init__(arms)
        self.epsilon = epsilon
        self.N = [0] * len(arms)
        self.Q = [0.0] * len(arms)

    def select_arm(self) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, len(self.arms) - 1)
        else:
            return self.Q.index(max(self.Q))

    def update(self, arm: int, reward: float) -> None:
        self.N[arm] += 1
        self.Q[arm] += (1 / self.N[arm]) * (reward - self.Q[arm])


class UCB(Policy):
    def __init__(self, arms: list[Arm]):
        super().__init__(arms)
        self.N = [0] * len(arms)
        self.Q = [0.0] * len(arms)
        self.QU = [0.0] * len(arms)

    def select_arm(self) -> int:
        return self.QU.index(max(self.QU))

    def update(self, arm: int, reward: float) -> None:
        self.N[arm] += 1
        self.Q[arm] += (1 / self.N[arm]) * (reward - self.Q[arm])
        for i in range(len(self.QU)):
            self.QU[i] = self.Q[i] + math.sqrt(
                math.log(sum(self.N)) / (2 * self.N[i] + 1)
            )


class ThompsonSampling(Policy):
    def __init__(self, arms: list[Arm]):
        super().__init__(arms)
        self.N = [0] * len(arms)
        self.S = [0] * len(arms)
        self.F = [0] * len(arms)

    def select_arm(self) -> int:
        self.Q = [0.0] * len(self.arms)
        for i in range(len(self.arms)):
            self.Q[i] = random.betavariate(self.S[i] + 1, self.F[i] + 1)
        return self.Q.index(max(self.Q))

    def update(self, arm: int, reward: float) -> None:
        self.N[arm] += 1
        if reward == 1:
            self.S[arm] += 1
        else:
            self.F[arm] += 1
        self.S[arm] = self.S[arm] + 1
        self.F[arm] = self.F[arm] + 1


def simulate(policy: Policy, arms: list[Arm], num_trials: int, num_time_steps: int):
    rewards = []
    for _ in range(num_trials):
        rewards.append([])
        for _ in range(num_time_steps):
            arm = policy.select_arm()
            reward = arms[arm].pull()
            policy.update(arm, reward)
            rewards[-1].append(reward)
    return rewards


def main():
    reward_data = []
    for i in range(1, 10):
        for j in range(1, 10):
            arms = [Arm(i / 10), Arm(j / 10)]
            policies = [
                Greedy(arms),
                EpsilonGreedy(arms, 0.1),
                UCB(arms),
                ThompsonSampling(arms),
            ]
            for policy in policies:
                rewards = simulate(policy, arms, 5, 100)
                reward_sum = [sum(i) for i in rewards]

                reward_data.append(
                    {
                        "policy": policy.__class__.__name__,
                        "left_arm": arms[0].p,
                        "right_arm": arms[1].p,
                        "rewards": sum(reward_sum) / len(reward_sum),
                    }
                )

    reward_dataframe = pd.DataFrame(reward_data)
    reward_dataframe.to_csv("files/2_reward_data.csv", index=None)


if __name__ == "__main__":
    main()
