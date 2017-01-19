import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import numpy as np
from random import randint

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(
            env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.reset()
        self.matQ = np.ones((512, 4))
        self.learningRate = 0.50
        self.exploration = 0.0
        self.discount = 0.01
        self.reward = 0
        self.aggReward = 0
        self.steps = 0


    def reset(self, destination=None):
        self.planner.route_to(destination)
        self.state = {
            'next_waypoint': None,
            'light': 'green',
            'oncoming': None,
            'right': None,
            'left': None
        }
        self.last_action = None
        self.aggReward = 0
        self.steps = 0

    def makeIndex(self, state):
        valid_light = ['green', 'red']
        lightIdx = valid_light.index(state['light'])
        wpIdx = Environment.valid_actions.index(state['next_waypoint'])
        onIdx = Environment.valid_actions.index(state['oncoming'])
        rIdx = Environment.valid_actions.index(state['right'])
        lIdx = Environment.valid_actions.index(state['left'])

        index = lightIdx * 4 * 4 * 4 * 4 + wpIdx * 4 * 4 * 4 + onIdx * 4 * 4 + rIdx * 4 + lIdx
        return index

    def bestAction(self, idx):

        #ugly but working :)
        actions = {}
        for action in Environment.valid_actions:
            actions[action] = self.matQ[idx, Environment.valid_actions.index(action)]
        # explore
        if random.uniform(0.0, 1.0) < self.exploration:
            rnd = Environment.valid_actions[randint(0, 3)]
            return rnd, actions[rnd]
        if actions['left'] == actions['right'] == actions['forward'] == actions[None]:
            rnd = Environment.valid_actions[randint(0, 3)]
            return rnd, actions[rnd]
        import operator
        desc = sorted(actions.items(), key=operator.itemgetter(1), reverse=True)
        return desc[0]

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        # previous state
        previous_state = self.state
        previous_action = self.last_action
        previous_reward = self.reward
        index_previous_state= self.makeIndex(previous_state)
        index_previous_action = Environment.valid_actions.index(previous_action)

        # TODO: Update state

        self.state = {
            'next_waypoint': self.next_waypoint,
            'light': inputs['light'],
            'oncoming': inputs['oncoming'],
            'right': inputs['right'],
            'left': inputs['left']
        }
        current_index = self.makeIndex(self.state)

        # TODO: Select action according to your policy

        action, bestQ = self.bestAction(current_index)
        # Execute action and get reward
        self.reward = self.env.act(self, action)
        self.aggReward += self.reward
        self.steps += 1
        self.last_action = action

        if (self.state['next_waypoint'] != action):
            print 'smartcab does not take {} but {}: '.format(self.state['next_waypoint'] , action)
            print 'lights are ' + self.state['light']

        # TODO: Learn policy based on state, action, reward
        self.matQ[index_previous_state, index_previous_action] = (1 - self.learningRate) * self.matQ[index_previous_state, index_previous_action] \
                                                                 + self.learningRate * (previous_reward + self.discount* bestQ)


        print "state= {}, reward = {}, action = {}, QValue = {}".format(index_previous_state, previous_reward, previous_action, self.matQ[index_previous_state, index_previous_action])  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent

    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.2, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
