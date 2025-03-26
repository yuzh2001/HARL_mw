import copy
import logging
import supersuit as ss
# from pettingzoo.sisl import multiwalker_v9
from .walker.multiwalker_stable import parallel_env as multiwalker_stable_parallel_env

logging.basicConfig()
logging.getLogger().setLevel(logging.ERROR)


class PettingZooMWEnv:
    def __init__(self, args):
        self.args = copy.deepcopy(args)
        self.discrete = False
        if "max_cycles" in self.args:
            self.max_cycles = self.args["max_cycles"]
            self.args["max_cycles"] += 1
        else:
            self.max_cycles = 500
            self.args["max_cycles"] = 501
        self.cur_step = 0
        # self.module = multiwalker_v9
        self.base_env = multiwalker_stable_parallel_env(**self.args)
        self.env = ss.pad_action_space_v0(
            ss.pad_observations_v0(self.base_env)
        )
        self.env.reset()
        self.n_agents = self.env.num_agents
        self.agents = self.env.agents
        self.share_observation_space = self.repeat(self.env.state_space)
        self.observation_space = self.unwrap({agent: self.env.observation_space(agent) for agent in self.agents})
        self.action_space = self.unwrap({agent: self.env.action_space(agent) for agent in self.agents})
        self._seed = 0

    def step(self, actions):
        """
        return local_obs, global_state, rewards, dones, infos, available_actions
        """
        obs, rew, term, trunc, info = self.env.step(self.wrap(actions))
        self.cur_step += 1
        if self.cur_step == self.max_cycles:
            trunc = {agent: True for agent in self.agents}
            for agent in self.agents:
                info[agent]["bad_transition"] = True
        dones = {agent: term[agent] or trunc[agent] for agent in self.agents}
        s_obs = self.repeat(self.env.state())
        total_reward = sum([rew[agent] for agent in self.agents])
        rewards = [[total_reward]] * self.n_agents
        return (
            self.unwrap(obs),
            s_obs,
            rewards,
            self.unwrap(dones),
            self.unwrap(info),
            self.get_avail_actions(),
        )

    def reset(self):
        """Returns initial observations and states"""
        self._seed += 1
        self.cur_step = 0
        obs, infos = self.env.reset(seed=self._seed)
        obs = self.unwrap(obs)
        s_obs = self.repeat(self.env.state())
        return obs, s_obs, self.get_avail_actions()

    def get_avail_actions(self):
        if self.discrete:
            avail_actions = []
            for agent_id in range(self.n_agents):
                avail_agent = self.get_avail_agent_actions(agent_id)
                avail_actions.append(avail_agent)
            return avail_actions
        else:
            return None

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id"""
        return [1] * self.action_space[agent_id].n

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def seed(self, seed):
        self._seed = seed

    def wrap(self, lam):
        d = {}
        for i, agent in enumerate(self.agents):
            d[agent] = lam[i]
        return d

    def unwrap(self, d):
        _tmp = []
        for agent in self.agents:
            _tmp.append(d[agent])
        return _tmp

    def repeat(self, a):
        return [a for _ in range(self.n_agents)]
