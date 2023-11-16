from pydantic import BaseModel


class Problem(BaseModel):
    name: str
    map: list[list[int]]
    actions: list[list[int]]
    start: list[tuple[int, int]]
    end: list[tuple[int, int]]
    reward_per_step: int


def print_value_map(index: int, value_map: list[list[float]]):
    print(f"======== {index} ========")
    for row in value_map:
        print([round(i, 3) for i in row])
    print()


def update_value_for_policy(problem: Problem, value_map: list[list[float]]):
    new_value_map = [
        [0 for _ in range(len(problem.map[0]))] for _ in range(len(problem.map))
    ]

    for i in range(len(problem.map)):
        for j in range(len(problem.map[0])):
            if (i, j) in problem.end or problem.map[i][j] == 0:
                new_value_map[i][j] = 0
            else:
                value_list = []
                for action in problem.actions:
                    next_i = i + action[0]
                    next_j = j + action[1]
                    if (
                        next_i < 0
                        or next_i >= len(problem.map)
                        or next_j < 0
                        or next_j >= len(problem.map[0])
                    ):
                        value_list.append(value_map[i][j] + problem.reward_per_step)
                    elif problem.map[next_i][next_j] == 1:
                        value_list.append(
                            value_map[next_i][next_j] + problem.reward_per_step
                        )
                # print(value_list)

                new_value_map[i][j] = sum(value_list) / len(value_list)
    return new_value_map


def update_value(problem: Problem, value_map: list[list[float]]):
    new_value_map = [
        [0.0 for _ in range(len(problem.map[0]))] for _ in range(len(problem.map))
    ]

    for i in range(len(problem.map)):
        for j in range(len(problem.map[0])):
            if (i, j) in problem.end or problem.map[i][j] == 0:
                new_value_map[i][j] = 0
            else:
                value_list = []
                for action in problem.actions:
                    next_i = i + action[0]
                    next_j = j + action[1]
                    if (
                        next_i < 0
                        or next_i >= len(problem.map)
                        or next_j < 0
                        or next_j >= len(problem.map[0])
                    ):
                        value_list.append(value_map[i][j] + problem.reward_per_step)
                    elif problem.map[next_i][next_j] == 1:
                        value_list.append(
                            value_map[next_i][next_j] + problem.reward_per_step
                        )
                new_value_map[i][j] = max(value_list)
    return new_value_map


def diff_value(map: list[list[int]], new_value_map: list[list[float]]):
    diff = 0
    for i in range(len(map)):
        for j in range(len(map[0])):
            diff += abs(map[i][j] - new_value_map[i][j])
    return diff


def draw_value_map(name: str, problem: Problem, value_map: list[list[float]]):
    import plotly.express as px

    fig = px.imshow(value_map)
    fig.update_layout(
        title=f"{name}-{problem.name}",
        xaxis_title="x",
        yaxis_title="y",
        font=dict(size=18),
    )
    for i, j in problem.start:
        fig.add_annotation(
            x=j,
            y=i,
            text="start",
            showarrow=False,
            font=dict(size=18),
        )
    for i, j in problem.end:
        fig.add_annotation(
            x=j,
            y=i,
            text="end",
            showarrow=False,
            font=dict(size=18),
        )

    # save image
    fig.write_image(f"images/3_{name}-{problem.name}.png")


def policy_evaluation(problem: Problem):
    value_map = [
        [0 for _ in range(len(problem.map[0]))] for _ in range(len(problem.map))
    ]
    i = 0
    while True:
        new_value_map = update_value_for_policy(problem, value_map)
        print_value_map(i, value_map)
        if diff_value(value_map, new_value_map) < 0.001:
            print(f"Converge at {i} iteration")
            break
        value_map = new_value_map
        i += 1

    draw_value_map("PolicyEvaluation", problem, value_map)


def value_iteration(problem: Problem):
    value_map = [
        [0.0 for _ in range(len(problem.map[0]))] for _ in range(len(problem.map))
    ]
    i = 0
    while True:
        new_value_map = update_value(problem, value_map)
        print_value_map(i, value_map)
        if diff_value(value_map, new_value_map) < 0.001:
            print(f"Converge at {i} iteration")
            break
        value_map = new_value_map
        i += 1

    draw_value_map("ValueIteration", problem, value_map)


if __name__ == "__main__":
    easy = Problem(
        name="easy",
        map=[[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
        actions=[[0, 1], [0, -1], [1, 0], [-1, 0]],
        start=[],
        end=[(0, 0), (3, 3)],
        reward_per_step=-1,
    )

    hard = Problem(
        name="hard",
        map=[
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 0, 0, 1, 0, 1, 0],
            [0, 1, 1, 0, 0, 1, 1, 0],
            [0, 0, 1, 1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0, 1, 1, 0],
            [0, 1, 1, 1, 1, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
        actions=[
            [0, 1],
            [0, -1],
            [1, 0],
            [-1, 0],
        ],
        start=[(2, 0)],
        end=[(6, 7)],
        reward_per_step=-1,
    )
    policy_evaluation(easy)
    value_iteration(easy)
    policy_evaluation(hard)
    value_iteration(hard)
