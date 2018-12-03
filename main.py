import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=3, suppress=True)


def allow(number):
    if number == 0:
        return "↑"
    elif number == 1:
        return "→"
    elif number == 2:
        return "↓"
    else:
        return "←"

def action_to_allow(sa_history):
    for i in range(len(sa_history)):
        sa_history[i][1] = allow(sa_history[i][1])
    return sa_history

# パラメータから方策を求める関数
def policy(theta):
    [m, n] = theta.shape
    pi = np.zeros((m, n))

    for i in range(0, m):
        pi[i, :] = theta[i, :] / np.nansum(theta[i, :])

    pi = np.nan_to_num(pi)
    return pi

# 行動aと次の状態sを求める関数
def action_and_next_s(pi, s):
    direction = ["up", "right", "down", "left"]
    next_direction = np.random.choice(direction, p=pi[s, :])

    if next_direction == "up":
        action = 0
        s_next = s - 4
    elif next_direction == "right":
        action = 1
        s_next = s + 1
    elif next_direction == "down":
        action = 2
        s_next = s + 4
    elif next_direction == "left":
        action = 3
        s_next = s - 1
    return [action, s_next]

# 迷路を解く関数
def solve(pi):
    s = 0  # スタート
    sa_history = [[0, np.nan]]
    while (1):
        [action, next_s] = action_and_next_s(pi, s)
        sa_history[-1][1] = action
        sa_history.append([next_s, np.nan])

        if next_s == 15:  #ゴール
            break
        else:
            s = next_s
    return sa_history

# パラメータを更新する関数
def update_theta(theta, pi, sa_history):
    eta = 0.1 # 学習率
    T = len(sa_history) - 1  # ゴールまでの総ステップ数

    [m, n] = theta.shape
    delta_theta = theta.copy()

    for i in range(0, m):
        for j in range(0, n):
            if not(np.isnan(theta[i, j])):
                SA_i = [SA for SA in sa_history if SA[0] == i]
                SA_ij = [SA for SA in sa_history if SA == [i, j]]

                N_i = len(SA_i)
                N_ij = len(SA_ij)

                delta_theta[i, j] = (N_ij - pi[i, j] * N_i) / T

    new_theta = theta + eta * delta_theta
    return new_theta


def main():
    # 初期パラメータ
    theta_0 = np.array([[np.nan, 1, 1, np.nan],       # s0
                        [np.nan, 1, 1, 1],            # s1
                        [np.nan, np.nan, np.nan, 1],  # s2
                        [np.nan, np.nan, 1, np.nan],  # s3
                        [1, np.nan, 1, np.nan],       # s4
                        [1, 1, 1, np.nan],            # s5
                        [np.nan, 1, np.nan, 1],       # s6
                        [1, np.nan, np.nan, 1],       # s7
                        [1, np.nan, 1, np.nan],       # s8
                        [1, 1, np.nan, np.nan],       # s9
                        [np.nan, 1, np.nan, 1],       # s10
                        [np.nan, np.nan, 1, 1],       # s11
                        [1, 1, np.nan, np.nan],       # s12
                        [np.nan, 1, np.nan, 1],       # s13
                        [np.nan, np.nan, np.nan, 1],  # s14
                        ])

    # 初期の方策
    pi_0 = policy(theta_0)
    print('Initial policy：\n{}'.format(pi_0))

    # 初期の方策で迷路を解く
    sa_history = solve(pi_0)
    print(action_to_allow(sa_history))
    print("steps : " + str(len(sa_history) - 1))

    # 方策勾配法で迷路を解く
    stop_epsilon = 10**-4

    theta = theta_0
    pi = pi_0

    is_continue = True
    count = 0
    steps = []
    while is_continue:
        count += 1
        sa_history = solve(pi)
        new_theta = update_theta(theta, pi, sa_history)
        new_pi = policy(new_theta)

        if count % 30 == 0:
            steps.append(len(sa_history) - 1)
            print("Update amount{:.4f}  Steps {:}".format(np.sum(np.abs(new_pi - pi)), len(sa_history) - 1))

        if np.sum(np.abs(new_pi - pi)) < stop_epsilon:
            is_continue = False
        else:
            theta = new_theta
            pi = new_pi

    # ステップ数の変化をグラフに
    plt.plot(steps)
    plt.ylabel("steps")
    plt.show()

    # 最終的な方策を確認
    print(pi)

    # 最終的な方策で迷路を解く
    sa_history = solve(pi)
    print(action_to_allow(sa_history))
    print("steps : " + str(len(sa_history) - 1))

if __name__ == '__main__':
    main()
