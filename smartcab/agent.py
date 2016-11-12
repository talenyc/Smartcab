from __future__ import print_function, division
import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import numpy as np

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.actions = [None, 'forward', 'left', 'right']
        self.learning_rate = 0.3
        self.state = None
        self.q = {}
        self.trips = 0

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.planner.route_to(destination)
        # Prepare for a new trip; reset any variables here, if required
        self.trips += 1
        if self.trips >= 100:
            self.learning_rate = 0

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        #state=None
                # Update state
        state = (inputs['light'], inputs['oncoming'], inputs['right'], inputs['left'],
self.next_waypoint)
        self.state = state
        

        for action in self.actions:
            if (state, action) not in self.q:
                self.q[(state, action)] = 1


        """"# TODO: Select action according to your policy
        #print 'valid_actions', random.choice(Environment.valid_actions[1:])
        action_okay = True
        if self.next_waypoint == 'right':
            if inputs['light'] == 'red' and inputs['left'] == 'forward':
                action_okay = False
        elif self.next_waypoint == 'forward':
            if inputs['light'] == 'red':
                action_okay = False
        elif self.next_waypoint == 'left':
            if inputs['light'] == 'red' or (inputs['oncoming'] == 'forward' or inputs['oncoming'] == 'right'):
                action_okay = False

        action = None
        if action_okay:
            action = self.next_waypoint
            self.next_waypoint = random.choice(Environment.valid_actions[1:])
       """
        probabilities = [self.q[(state, None)], self.q[(state, 'forward')], self.q[(state, 'left')], self.q[(state, 'right')]]
        
       #  Q lerning =  Reward optimization based on Sate, Action and reward
        action = self.actions[np.argmax(probabilities)]
        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward
        
        #Learn policy based on state, action, reward
        self.q[(state, action)] = self.learning_rate * reward + (1 - self.learning_rate) * self.q[(state, action)]

    def get_state(self):
        return self.state

        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
        #print Q_table


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=False)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=1.0)  # reduce update_delay to speed up simulation
    sim.run(n_trials=1)  # press Esc or close pygame window to quit


if __name__ == '__main__':
    run()
