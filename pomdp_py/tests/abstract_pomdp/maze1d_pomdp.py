import numpy as np
import random
import math
import sys
import time
import copy
from pomdp_py import *


# Unit testing
class Maze1D_State(State):
    def __init__(self, robot_pose, target_pose):
        self.robot_pose = robot_pose
        self.target_pose = target_pose
        super().__init__(data=[])  # data is not useful if hash function is overridden.
    def unpack(self):
        return self.robot_pose, self.target_pose
    def __str__(self):
        return 'Maze1D_State::[%s, %s]' % (str(self.robot_pose), str(self.target_pose))
    def __repr__(self):
        return self.__str__()
    def __eq__(self, other):
        return self.target_pose == other.target_pose\
            and self.robot_pose == other.robot_pose
    def __hash__(self):
        return hash((self.robot_pose, self.target_pose))
    def __getitem__(self, index):
        raise NotImplemented
    def __len__(self):
        raise NotImplemented

class Maze1D_BeliefState(BeliefState):
    # This is a belief state regarding a SINGLE target, or a robot.
    def __init__(self, distribution, name="BeliefState"):
        super().__init__(distribution, name=name)

class Maze1D(Environment):
    def __init__(self, worldstr):
        """worldstr: e.g. ...R...T.."""
        self.worldstr = worldstr
        self._d = np.zeros((len(worldstr),))
        for i, c in enumerate(worldstr):
            if c == ".":
                self._d[i] = 0
            elif c == "T":
                self._d[i] = 1
                target_pose = i
            elif c == "R":
                self._d[i] = 0
                robot_pose = i
        self.cur_state = Maze1D_State(robot_pose, target_pose)
        
    @property
    def robot_pose(self):
        return self.cur_state.robot_pose
    
    @property
    def target_pose(self):
        return self.cur_state.target_pose

    @property
    def state(self):
        return self.cur_state

    def state_transition(self, action, execute=True):
        """The maze environment only understands low-level actions."""
        next_state = Maze1D_State(self.if_move_by(self.robot_pose, action),
                                  self.target_pose)
        if execute:
            self.cur_state = next_state
        return next_state
    
    def move_robot(self, action):
        self._robot_pose = self.if_move_by(self.robot_pose, action)
        
    def if_move_by(self, robot_pose, movement):
        return max(0, min(len(self._d), robot_pose + movement))
    
    def __len__(self):
        return len(self._d)
    
    def if_move_by(self, robot_pose, action, world_range=None):
        if type(action) != int:
            return robot_pose
        if world_range is None:
            world_range = (0, len(self._d))
        return max(world_range[0], min(world_range[1]-1, robot_pose + action))
    
    def __len__(self):
        return len(self._d)


## WARNING. POMDP will not maintain true state. True state should be maintained by
# some other object (the environment).
    
class Maze1DPOMDP(POMDP):
    def __init__(self, maze, gamma=0.99, prior="RANDOM", representation="particles", **kwargs):
        self.maze = maze
        world_range = kwargs.get("world_range", (0, len(self.maze)))  #inclusive, exclusive
        world_range = max(0, world_range[0]), min(len(self.maze), world_range[1])
        self.world_range = world_range

        if representation == "exact":
            histogram = {}
            total_prob = 0
            for i in range(*world_range):
                state = Maze1D_State(self.robot_pose, i)
                if prior == "GROUNDTRUTH":
                    if i == self.target_pose:
                        histogram[state] = 0.8
                    else:
                        histogram[state] = 0.2
                else:  # RANDOM
                    histogram[state] = 1.0
                total_prob += histogram[state]
            for state in histogram:
                histogram[state] /= total_prob
            init_belief_distribution = BeliefDistribution_Histogram(histogram)
            init_belief = Maze1D_BeliefState(init_belief_distribution)
        elif representation == "particles":
            num_particles = kwargs.get("num_particles", 1000)
            particles = []
            if prior == "RANDOM":
                num_per_cell = num_particles // len(self.maze)
                for i in range(*world_range):
                    for _ in range(num_per_cell):
                        particles.append(Maze1D_State(self.robot_pose, i))
            else: #GROUNDTRUTH
                while len(particles) < num_particles:
                    particles.append(Maze1D_State(self.robot_pose, self.target_pose))
            init_belief_distribution = POMCP_Particles(particles)
            init_belief = Maze1D_BeliefState(init_belief_distribution)
        actions = [1, -1, AbstractPOMDP.BACKTRACK]
        self._last_real_action = None
        POMDP.__init__(self, actions, self.transition_func, self.reward_func, self.observation_func,
                       init_belief, None, gamma=gamma)
    @property
    def robot_pose(self):
        return max(0, self.world_range[0], min(self.maze.robot_pose, self.world_range[1]))
    @property
    def target_pose(self):
        return self.maze.target_pose
    @property
    def world(self):
        return self.maze

    def transition_func(self, state, action):
        next_robot_pose = self.maze.if_move_by(state.robot_pose, action, world_range=self.world_range)
        next_state = Maze1D_State(next_robot_pose,
                                  state.target_pose)
        return next_state

    def observation_func(self, next_state, action):
        if abs(next_state.robot_pose - next_state.target_pose) <= 0:
            return next_state.target_pose
        return -1        

    def reward_func(self, state, action, next_state):
        cur_robot_pose = state.robot_pose
        next_robot_pose = next_state.robot_pose
        reward = 0
        if cur_robot_pose == next_robot_pose:
            reward -= 10  # didn't move sucessfully
        if self.maze.target_pose < self.world_range[0] or self.maze.target_pose >= self.world_range[1]:
            if action == AbstractPOMDP.BACKTRACK:
                reward += 10
        else:
            if action == AbstractPOMDP.BACKTRACK:
                reward -= 10
            else:
                reward += 10*math.exp(-abs(next_state.robot_pose - self.maze.target_pose))
        return reward - 0.01
    
    def belief_update(self, real_action, real_observation, **kwargs):
        print("updating belief (concrete pomdp)>>>>")
        # print(self.cur_belief.distribution.mpe())
        self.cur_belief.update(real_action, real_observation,
                               self, **kwargs)
        print(self.cur_belief)
        print(">>>>")

    def is_in_goal_state(self):
        return self.cur_belief.distribution.mpe().robot_pose == self.maze.target_pose\
            or self._last_real_action == AbstractPOMDP.BACKTRACK

    def add_transform(self, state):
        """Used for particle reinvigoration"""
        state.target_pose = max(0, min(len(self.maze)-1, state.target_pose + random.randint(-1, 1)))

    def print_true_state(self):
        s = ["."] * len(self.maze)
        s[self.robot_pose] = "R"
        s[self.target_pose] = "T"
        print("".join(s))


class POMDPExperiment:

    def __init__(self, maze, pomdp, planner, max_episodes=100):
        self._planner = planner
        self._maze = maze  # the environment
        self._pomdp = pomdp
        self._discounted_sum_rewards = 0
        self._num_iter = 0
        self._max_episodes = max_episodes

    def run(self):
        # self._env.on_loop()
        self._num_iter = 0
        total_time = 0
        rewards = []
        try:
            while not self._pomdp.is_in_goal_state()\
                  and (self._num_iter < self._max_episodes):

                reward = None

                start_time = time.time()
                action = self._planner.plan_next_action()
                total_time += time.time() - start_time

                state = copy.deepcopy(self._maze.state)
                next_state = self._maze.state_transition(action)
                observation, reward = self._pomdp.real_action_taken(action, state, next_state)
                self._pomdp.belief_update(action, observation, **self._planner.params)
                self._planner.update(action, observation)
                
                # action, reward, observation = \
                #     self._planner.plan_and_execute_next_action()  # the action is a control to the robot
                
                if reward is not None:
                    self._pomdp.print_true_state()                
                    print("---------------------------------------------")
                    print("%d: Action: %s; Reward: %.3f; Observation: %s"
                          % (self._num_iter, str(action), reward, str(observation)))
                    print("---------------------------------------------")
                    self._discounted_sum_rewards += ((self._planner.gamma ** self._num_iter) * reward)
                    rewards.append(reward)

                # self._env._last_observation = self._pomdp.gridworld.provide_observation()
                self._num_iter += 1
        except KeyboardInterrupt:
            print("Stopped")
            return

        print("Done!")
        return total_time, rewards

    @property
    def discounted_sum_rewards(self):
        return self._discounted_sum_rewards

    @property
    def num_episode(self):
        return self._num_iter

def _rollout_policy(tree, actions):
    return random.choice(actions)    
                   
def unittest():
    num_particles = 1000
    maze = Maze1D(sys.argv[1])
    pomdp = Maze1DPOMDP(maze, prior="RANDOM", representation="particles",
                        num_particles=num_particles)
    planner = POMCP(pomdp, num_particles=num_particles,
                    max_time=1.0, max_depth=100, gamma=0.6, rollout_policy=_rollout_policy,
                    exploration_const=math.sqrt(2))
    
    experiment = POMDPExperiment(maze, pomdp, planner, max_episodes=100)
    experiment.run()

if __name__ == '__main__':
    unittest()
