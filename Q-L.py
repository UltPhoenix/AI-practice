import gym
import numpy as np 
# 生成仿真环境
env = gym.make('Taxi-v3')
# 重置仿真环境
obs = env.reset()
# 渲染环境当前状态
env.render()
m = env.observation_space.n  # size of the state space
n = env.action_space.n  # size of action space
print("出租车问题状态数量为{:d}，动作数量为{:d}。".format(m, n))


def arg_max(state_action):
    max_index_list=[]
    max_value=state_action[0]
    for index,value in enumerate(state_action):
        if value>max_value:
            max_index_list.clear()
            max_value=value
            max_index_list.append(index)
        elif value==max_value:
            max_index_list.append(index)
    return np.random.choice(max_index_list)
# Intialize the Q-table and hyperparameters
# Q表，大小为 m*n
Q = np.zeros([m,n])
# 回报的折扣率
gamma = 0.97
# 分幕式训练中最大幕数
max_episode = 1000
# 每一幕最长步数
max_steps = 100
# 学习率参数
alpha = 0.7# 随机探索概率
epsilon = 0.25

for i in range(max_episode):
    # Start with new environment
    s = env.reset()
    done = False
    counter = 0
    for j in range(max_steps):
        # Choose an action using epsilon greedy policy
        p = np.random.rand()
        # 请根据 epsilon-贪婪算法 选择动作 a
        # p > epsilon 或尚未学习到某个状态的价值时，随机探索
        if p>epsilon or not np.any(Q[s]):
            action=np.random.choice(n)
        else:
            st=Q[s]
            action=arg_max(st)
        # 其它情况，利用已经觉得的价值函数进行贪婪选择 (np.argmax)
        # ======= 将代码补充到这里

        # ======= 补充代码结束
        
        #根据所选动作action执行一步
        # 返回新的状态、回报、以及是否完成
        s_new, r, done, _ = env.step(action)
        new_q=r+gamma*np.max(Q[s_new])
        Q[s,action]=(1-alpha)*Q[s,action]+alpha*(new_q-Q[s,action])# 请根据贝尔曼方程，更新Q表 (np.max)
        # ======= 将代码补充到这里
        # ======= 补充代码结束
        print(Q[s,action],r)
        s = s_new
        if done:
            break

s = env.reset()
done = False
env.render()
# Test the learned Agent
rewards = []
for i in range(max_steps):
    a = np.argmax(Q[s,:])
    s, _, done, _ = env.step(a)
    rewards.append(Q[s,a])
    if done:
        break
# ======= 将代码补充到这里
rewards=np.array(rewards)
r_mean =rewards.mean()
r_var =rewards.var()
# ======= 补充代码结束

print("平均回报为{}，回报的方差为{}。".format(r_mean, r_var))
env.close()