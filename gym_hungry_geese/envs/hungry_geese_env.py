from kaggle_environments import make, evaluate
import gym
from gym import spaces
import numpy as np

# single frame observation
FOOD_OBS = 1.
FOOD_STEP = .004  # food value varies based on the hunger rate
NEXT_STEP_HUNGER = 0.7  # geese shrink every hunger_rate steps
geese_dict = {'head': -1., 'mid': -0.75, 'tail': -.5, 'last_head': -0.75}
# simplifying it a bit. might lose a little info in some edge cases
# geese_dict = {'agent': {'head': 0.75, 'mid': 0.375, 'tail': 0.25, 'last_head': 0.5, 'last_head_no_tail': 0.1},
#               0: {'head': -0.75, 'mid': -0.375, 'tail': -0.25, 'last_head': -0.5, 'last_head_no_tail': -0.1},
#               1: {'head': -0.75, 'mid': -0.375, 'tail': -0.25, 'last_head': -0.5, 'last_head_no_tail': -0.1},
#               2: {'head': -0.75, 'mid': -0.375, 'tail': -0.25, 'last_head': -0.5, 'last_head_no_tail': -0.1}}

# obs_list is either of length 1 on a reset (current_obs) or of length 2 (last obs, current obs)
# obs example: {'remainingOverageTime': 60, 'step': 1, 'geese': [[39], [32], [40], [60]], 'food': [24, 53], 'index': 0}
# index is the current agent's index in the list of geese
# head is always first


# turn kaggle into multi layer obs for agent to read
    # layer 0 is agent goose, 1 is agent prior head
    # layers 2-7 are same for opponents
    # layer 8 is food
def process_obs(obs_list, rows=7, columns=11, hunger_rate=40):
    # only need last two observations
    if len(obs_list) > 2:
        obs_list = obs_list[1:]

    obs_len = len(obs_list)
    new_obs = np.zeros((9, rows * columns), dtype=np.float32)

    # place food. I modify the food by the hunger rate so agent can hopefully predict when it will shrink
    hunger_steps = obs_list[-1]['step'] % hunger_rate
    if hunger_steps == hunger_rate - 1:
        food_value = NEXT_STEP_HUNGER
    else:
        food_value = FOOD_OBS - hunger_steps * FOOD_STEP
    for i in obs_list[-1]['food']:
        new_obs[8, i] = food_value

    # place geese and gees prior head
    geese_obs = obs_list[-1]['geese']
    agent_index = obs_list[-1]['index']
    # place geese body parts
    for i in range(0, len(geese_obs)):
        if i == agent_index:
            # agent always key 0
            geese_key = 0
        else:
            # opponent keys are 2, 4, 6
            if i < agent_index:
                geese_key = (i + 1)*2
            else:
                geese_key = i*2

        goose_len = len(geese_obs[i])
        # check if goose is alive
        if goose_len > 0:
            # place head
            new_obs[geese_key, geese_obs[i][0]] = geese_dict['head']
            # check if goose has other body parts than head
            if goose_len > 1:
                for j in range(1, goose_len - 1):
                    new_obs[geese_key, geese_obs[i][j]] = geese_dict['mid']
                # place tail
                new_obs[geese_key, geese_obs[i][-1]] = geese_dict['tail']
            # place head position from last turn
            if obs_len > 1:
                new_obs[geese_key+1, obs_list[0]['geese'][i][0]] = geese_dict['last_head']

    #print("debugging obs")
    #print(new_obs.reshape(9, rows, columns))
    # print(new_obs[6].reshape(1, rows, columns))
    # print(new_obs[7].reshape(1, rows, columns))

    # print("existing obs list", obs_list)
    # reshape new obs to grid
    return new_obs.reshape(9, rows, columns), obs_list


# turn kaggle env obs into 2D grid with everything represented
def process_obs_2D(obs_list, center_head=False, rows=7, columns=11, hunger_rate=40):
    # https://www.kaggle.com/victordelafuente/dqn-goose-with-stable-baselines3-pytorch
    # https://www.kaggle.com/yuricat/smart-geese-trained-by-reinforcement-learning

    # single frame observation
    FOOD_OBS = 1.
    FOOD_STEP = .001  # food value varies based on the hunger rate
    NEXT_STEP_HUNGER = 0.9  # geese shrink every hunger_rate steps
    geese_dict = {'agent': {'head': 0.325, 'mid': 0.1375, 'tail': 0.075, 'last_head': 0.2, 'last_head_no_tail': 0.2625},
                  0: {'head': -0.075, 'mid': -0.2625, 'tail': -0.325, 'last_head': -0.2, 'last_head_no_tail': -0.1375},
                  1: {'head': -0.4, 'mid': -0.5875, 'tail': -0.65, 'last_head': -0.525, 'last_head_no_tail': -0.4625},
                  2: {'head': -0.725, 'mid': -0.9125, 'tail': -0.975, 'last_head': -0.85, 'last_head_no_tail': -0.7875}}

    # only need last two observations
    if len(obs_list) > 2:
        obs_list = obs_list[1:]

    current_obs = obs_list[-1]
    obs_len = len(obs_list)
    new_obs = np.zeros((rows * columns), dtype=np.float32)
    # print("in process obs ", obs_list, obs_len)

    # place food. I modify the food by the hunger rate so agent can hopefully predict when it will shrink
    hunger_steps = current_obs['step'] % hunger_rate
    if hunger_steps == hunger_rate - 1:
        food_value = NEXT_STEP_HUNGER
    else:
        food_value = FOOD_OBS - hunger_steps * FOOD_STEP
    for i in current_obs['food']:
        new_obs[i] = food_value

    geese_obs = current_obs['geese']
    agent_index = current_obs['index']
    # place snake body parts
    for i in range(0, len(geese_obs)):
        if i == agent_index:
            geese_key = 'agent'
        else:
            if i > agent_index:
                geese_key = i - 1
            else:
                geese_key = i
        goose_len = len(geese_obs[i])
        # check if goose is alive
        if goose_len > 0:
            # place head
            # import pdb; pdb.set_trace()
            new_obs[geese_obs[i][0]] = geese_dict[geese_key]['head']
            # check if goose has other body parts than head
            if goose_len > 1:
                for j in range(1, goose_len - 1):
                    new_obs[geese_obs[i][j]] = geese_dict[geese_key]['mid']
                # place tail
                new_obs[geese_obs[i][-1]] = geese_dict[geese_key]['tail']
                # place head position from last turn
                if obs_len > 1:
                    new_obs[obs_list[0]['geese'][i][0]] = geese_dict[geese_key]['last_head']
            else:
                # goose is of length 1, not turn one and and if space is empty then place prior head position
                if obs_len > 1 and new_obs[obs_list[0]['geese'][i][0]] == 0:
                    new_obs[obs_list[0]['geese'][i][0]] = geese_dict[geese_key]['last_head_no_tail']

    # move head of agent into center position
    # if center_head:
    # new_obs = new_obs.reshape(rows, columns)
    # head_value = current_obs['geese'][agent_index][0]

    # print("debugging obs", obs_list[-1])
    # print(new_obs.reshape(rows, columns))

    # print("existings obs list", obs_list)
    # reshape new obs to grid
    return new_obs.reshape(rows, columns, 1), obs_list
    #return new_obs, obs_list


class HungryGeeseEnv(gym.Env):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, opponents=['greedy', 'greedy', 'greedy'], center_head=False, debug=False, episode_steps=200,
                 hunger_rate=40):
        super(HungryGeeseEnv, self).__init__()
        # self.num_envs = 1
        self.num_prev_obs = 1
        self.debug = debug
        ks_env = make("hungry_geese", configuration={'episodeSteps': episode_steps, 'hunger_rate': hunger_rate},
                      debug=debug)
        self.env = ks_env.train([None, *opponents])
        # config
        self.rows = ks_env.configuration.rows
        self.columns = ks_env.configuration.columns
        self.hunger_rate = ks_env.configuration.hunger_rate
        self.max_steps = ks_env.configuration.episodeSteps - 1  # not sure why it's minus 1, seems off
        self.agents = len(opponents) + 1
        # env takes strings only as actions ["NORTH", "EAST", "SOUTH", "WEST"]
        self.action_space = spaces.Discrete(4)
        self.action_list = ["NORTH", "EAST", "SOUTH", "WEST"]
        self.action_dict_opposite = {"NORTH": "SOUTH", "EAST": "WEST", "SOUTH": "NORTH", "WEST": "EAST", "NONE": "NONE"}
        self.last_action = "NONE"
        # reward
        self.reward_range = (-1, 1)  # not sure if this is needed
        self.reward_dict = {'survive': 0.0, 'food': 0.001, 'death': -1., 1: 1., 2: 0.1, 3: -0.8,
                            4: -0.9}  # the numbers are what place agent came in
        # obs_list stores obs from kaggle env. this env's obs space is preprocessed into a 2D grid of float values
        self.obs_list = []
        self.observation_space = spaces.Box(low=-1., high=1.,
                                            shape=(9, self.rows, self.columns), dtype=np.float32)
        # self.observation_space = spaces.Box(low=-1., high=1.,
        #                                     shape=(self.rows*self.columns,), dtype=np.float32)
        self.obs = self.env.reset()
        self.center_head = center_head  # center obs on agents head

    def step(self, action):
        action_string = self.action_list[action]
        # opposite action means instant death. if opposte action then choose random one among other 3
        if self.action_dict_opposite[self.last_action] == action_string:
            action = (action + np.random.randint(1,4)) % 4
            action_string = self.action_list[action]
        obs, r, done, info = self.env.step(action_string)
        self.obs_list.append(obs)
        reward = self._process_reward(r, self.obs_list, done)
        obs, self.obs_list = process_obs(self.obs_list, rows=self.rows, columns=self.columns,
                                         hunger_rate=self.hunger_rate)
        self.last_action = action_string
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        self.obs_list = [obs]
        obs, self.obs_list = process_obs(self.obs_list, rows=self.rows, columns=self.columns,
                                         hunger_rate=self.hunger_rate)
        self.last_action = "NONE"
        return obs

    # making this seperate function outside of the gym env because agent needs access to it at run time
    #     def _process_obs(self, obs):
    #         pass

    def _process_reward(self, r, obs_list, done):
        # default reward stream not normalized or ideal for RL agents, more like a 'score'
        # reward config is in self.reward_dict
        # tiebreaker is survival time then length apparently
        # https://www.kaggle.com/c/hungry-geese/discussion/214515
        if r == 0:
            # agent has 0 reward, thus must be eliminated
            return self.reward_dict['death']
        else:
            # agent has positive reward, thus must be alive
            if done:
                # three done conditions: agent eliminated (reward 0 above), opponents eliminated (else),
                # or env had reached max steps
                if obs_list[-1]['step'] == self.max_steps:
                    agent_index = obs_list[-1]['index']
                    goose_lengths = obs_list[-1]['geese']
                    train_agent_length = len(goose_lengths[agent_index])
                    # 2nd tiebreaker is final length I think
                    goose_place = 1
                    for i in range(0, self.agents):
                        if i == agent_index:
                            continue
                        if train_agent_length < len(goose_lengths[i]):
                            goose_place += 1
                    return self.reward_dict[goose_place]
                else:
                    # agent won
                    return self.reward_dict[1]
            else:
                # food bonus: compare lengths to see if food bonus
                if len(obs_list) > 1:
                    agent_index = obs_list[-1]['index']
                    current_length = len(obs_list[-1]['geese'][agent_index])
                    last_length = len(obs_list[-2]['geese'][agent_index])
                    if current_length > last_length:
                        return self.reward_dict['food']

                return self.reward_dict['survive']

    def render(self, mode='human'):
        print(self.obs_list[-1])
        obs, _ = process_obs_2D(self.obs_list, center_head=self.center_head, rows=self.rows, columns=self.columns,
                                hunger_rate=self.hunger_rate)
        print(obs.reshape(self.rows, self.columns))
