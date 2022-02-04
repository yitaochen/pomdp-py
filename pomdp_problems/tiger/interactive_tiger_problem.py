"""Tiger Problem with I-POMDP.

It builds on the classic Tiger Problem.

The description of the tiger problem with I-POMDP can be found in the following paper: (Quote from `A Framework
for Sequential Planning in Multi-Agent Settings
<https://arxiv.org/pdf/1109.2135.pdf>`_ by
Gmytrasiewicz and Doshi )

TBF

States: {tiger-left, pj}, {tiger-right, pj}
Actions: open-left, open-right, listen
Rewards:
    +10 for opening treasure door. -100 for opening tiger door.
    -1 for listening.
Observations: You can hear either "growl-left", or "growl-right" with the accuracy of 85%.

"""
import pomdp_py
from pomdp_py.utils import TreeDebugger
import random
import numpy as np
import sys

class TigerState(pomdp_py.State):
    def __init__(self, name):
        self.name = name
    def __hash__(self):
        return hash(self.name)
    def __eq__(self, other):
        if isinstance(other, TigerState):
            return self.name == other.name
        return False
    def __str__(self):
        return self.name
    def __repr__(self):
        return "TigerState(%s)" % self.name
    def other(self):
        if self.name.endswith("left"):
            return TigerState("tiger-right")
        else:
            return TigerState("tiger-left")

class TigerAction(pomdp_py.Action):
    def __init__(self, name):
        self.name = name
    def __hash__(self):
        return hash(self.name)
    def __eq__(self, other):
        if isinstance(other, TigerAction):
            return self.name == other.name
        return False
    def __str__(self):
        return self.name
    def __repr__(self):
        return "TigerAction(%s)" % self.name

class TigerObservation(pomdp_py.Observation):
    def __init__(self, name):
        self.name = name
    def __hash__(self):
        return hash(self.name)
    def __eq__(self, other):
        if isinstance(other, TigerObservation):
            return self.name == other.name
        return False
    def __str__(self):
        return self.name
    def __repr__(self):
        return "TigerObservation(%s)" % self.name

# Observation model
class ObservationModel(pomdp_py.ObservationModel):
    def __init__(self, noise=0.15):
        self.noise = noise

    def probability(self, observation, next_state, action):
        if action.name == "listen":
            if observation.name == next_state.name: # heard the correct growl
                return 1.0 - self.noise
            else:
                return self.noise
        else:
            return 0.5

    def sample(self, next_state, action):
        if action.name == "listen":
            thresh = 1.0 - self.noise
        else:
            thresh = 0.5

        if random.uniform(0,1) < thresh:
            return TigerObservation(next_state.name)
        else:
            return TigerObservation(next_state.other().name)

    def get_all_observations(self):
        """Only need to implement this if you're using
        a solver that needs to enumerate over the observation space (e.g. value iteration)"""
        return [TigerObservation(s) for s in {"tiger-left", "tiger-right"}]

# Transition Model
class TransitionModel(pomdp_py.TransitionModel):
    def probability(self, next_state, state, action):
        """According to problem spec, the world resets once
        action is open-left/open-right. Otherwise, stays the same"""
        if action.name.startswith("open"):
            return 0.5
        else:
            if next_state.name == state.name:
                return 1.0 - 1e-9
            else:
                return 1e-9

    def sample(self, state, action):
        if action.name.startswith("open"):
            return random.choice(self.get_all_states())
        else:
            return TigerState(state.name)

    def get_all_states(self):
        """Only need to implement this if you're using
        a solver that needs to enumerate over the observation space (e.g. value iteration)"""
        return [TigerState(s) for s in {"tiger-left", "tiger-right"}]

# Reward Model
class RewardModel(pomdp_py.RewardModel):
    def _reward_func(self, state, action):
        if action.name == "open-left":
            if state.name == "tiger-right":
                return 10
            else:
                return -100
        elif action.name == "open-right":
            if state.name == "tiger-left":
                return 10
            else:
                return -100
        else: # listen
            return -1

    def sample(self, state, action, next_state):
        # deterministic
        return self._reward_func(state, action)

# Reward Model for I-POMDP
class IRewardModel(pomdp_py.RewardModel):
    def _reward_func(self, istate, action):
        if action.name == "open-left":
            if istate.state().name == "tiger-right":
                return 10
            else:
                return -100
        elif action.name == "open-right":
            if istate.state().name == "tiger-left":
                return 10
            else:
                return -100
        else: # listen
            return -1

    def sample(self, istate, action, next_istate):
        # deterministic
        return self._reward_func(istate, action)
    
# Policy Model
class PolicyModel(pomdp_py.RandomRollout):
    """This is an extremely dumb policy model; To keep consistent
    with the framework."""
    # A stay action can be added to test that POMDP solver is
    # able to differentiate information gathering actions.
    ACTIONS = {TigerAction(s) for s in {"open-left", "open-right", "listen"}}

    def sample(self, state, **kwargs):
        return random.sample(self.get_all_actions(), 1)[0]

    def get_all_actions(self, **kwargs):
        return PolicyModel.ACTIONS

# Interactive State for Interactive Tiger Problem
class ITigerState(TigerState):

    def __init__(self, state, belief):
        self._state = state 
        self._belief = belief

    def __hash__(self):
        return hash(self._state)
    
    def __eq__(self, other):
        if isinstance(other, ITigerState):
            return self._state == other._state and self._belief == other._belief
        return False

    def __str__(self):
        return "(TigerState(%s), %s)" % (self._state.name, self._belief)

    def __repr__(self):
        # It could be more verbose as 
        # return "(TigerState(%s), {TigerState(tiger-left): %.4f, TigerState(tiger-right): %.4f})" % (self._state.name, self._belief[TigerState("tiger-left")], self._belief[TigerState("tiger-right")])
        # But for easy reading, I use the following alternative representation instead
        return "(TigerState(%s), %s)" % (self._state.name, self._belief)

    def state(self):
        return self._state
    
    def belief(self):
        return self._belief

# Belief for I-POMDP
class IBelief(pomdp_py.GenerativeDistribution):

    def __init__(self, beliefs):
        """
        beliefs (dict): maps from interactive-state (ITigerState) to a belief (GenerativeDistribution)
        """
        self._beliefs = beliefs

    def mpe(self):
        """Returns the most likely state"""
        return max(self._beliefs, key = self._beliefs.get)

    def random(self):
        """Samples a random state"""
        return random.sample(self._beliefs.items(), 1)[0][0]

    def __getitem__(self, istate):
        """Returns the probability of istate"""
        return self._beliefs[istate]

    def __iter__(self):
        return iter(self._beliefs)
    
    def __str__(self):
        if isinstance(self._beliefs, dict):
            # for istate, prob in self._beliefs.items():
            #     print(istate)
            return str(["%s:%.4f" % (istate, prob) for istate, prob in self._beliefs.items()])

# Interactive Tiger Problem
class ITigerProblem(pomdp_py.POMDP):
    """
    In fact, creating a TigerProblem class is entirely optional
    to simulate and solve POMDPs. But this is just an example
    of how such a class can be created.
    """

    def __init__(self, obs_noise, init_true_state, init_belief):
        """init_belief is a Distribution."""
        agent = pomdp_py.Agent(init_belief,
                               PolicyModel(),
                               TransitionModel(),
                               ObservationModel(obs_noise),
                               RewardModel())
        env = pomdp_py.Environment(init_true_state,
                                   TransitionModel(),
                                   RewardModel())
        super().__init__(agent, env, name="InteractiveTigerProblem")

    @staticmethod
    def create(state="tiger-left", belief=0.5, obs_noise=0.15):
        """
        Args:
            state (str): could be 'tiger-left' or 'tiger-right'; True state of the environment
            belief (float): Initial belief that the target is on the left; Between 0-1.
            obs_noise (float): Noise for the observation model (default 0.15)
        """
        init_true_state = TigerState(state)
        init_belief = pomdp_py.Histogram({TigerState("tiger-left"): 0.5,
                                        TigerState("tiger-right"): 0.5})                                                              
        tempDict = dict()
        tempDict[ITigerState(TigerState("tiger-left"), init_belief)] = 0.5
        tempDict[ITigerState(TigerState("tiger-right"), init_belief)] = 0.5
        init_i_belief = IBelief(tempDict)
        tiger_problem = ITigerProblem(obs_noise,  # observation noise
                                     init_true_state, init_i_belief)
        tiger_problem.agent.set_belief(init_i_belief, prior=True)
        return tiger_problem


def main():
    init_true_state = random.choice([TigerState("tiger-left"),
                                     TigerState("tiger-right")])
    init_belief = pomdp_py.Histogram({TigerState("tiger-left"): 0.5,
                                      TigerState("tiger-right"): 0.5})                                                              
    tempDict = dict()
    IST = ITigerState(TigerState("tiger-left"), init_belief)
    print(IST.state())
    tempDict[ITigerState(TigerState("tiger-left"), init_belief)] = 0.5
    tempDict[ITigerState(TigerState("tiger-right"), init_belief)] = 0.5
    init_i_belief = IBelief(tempDict)
    i_tiger_problem = ITigerProblem(0.15,  # observation noise
                                init_true_state, init_i_belief)
    

if __name__ == '__main__':
    main()





