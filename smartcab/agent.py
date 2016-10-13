from __future__ import print_function, division
import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import numpy as np
import pandas as pd

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
        self.Qtable = {}
        self.trips = 0

        self.Q_init = 0  #initial Q^ values for new state-actions not observed yet.
        
        self.gamma = .87
        #self.gamma = 0.1  #discounting rate of future rewards

        self.epsilon = 1
        #self.epsilon = 0.75 + (0.24 / (1+( math.exp(-0.1*(self.lesson_counter-40)))))
        
        self.alpha = 1
        #self.alpha = 1 - ( 0.5 / (1 + math.exp(-0.05*(self.lesson_counter-100)))) #alpha ranges from 1 to 0.5

        # logging
        self.learning_table = []

        self.reward_previous = None
        self.action_previous = None
        self.state_previous = None

    def reset(self, destination=None):
        #log trip and success rate 
        #
        #print("trip = {}, destination_reached = {}".format(self.trips, self.env.destination_reached))
        self.planner.route_to(destination)
        # Prepare for a new trip; reset any variables here, if required
        self.trips += 1
        self.steps_counter = 0

        

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
        
        # TODO: Select action according to your policy
        Qtable = self.Qtable #populate Qtable variable with current object.Qtable 
        if Qtable.has_key(self.state): #check if state has been encountered before or not
            if random.random() < self.epsilon : #if epsilon is less than  random float, then choose the action with the largest Q^
                argmax_actions = {action:Qhat for action, Qhat in Qtable[self.state].items() if Qhat == max(Qtable[self.state].values())}
                action = random.choice(argmax_actions.keys())
            else : # if random is greater, choose a random action.
                action = random.choice([None, 'forward', 'left', 'right'])
        else :  #state has never been encountered
            Qtable.update({self.state : {None : self.Q_init, 'forward' : self.Q_init, 'left' : self.Q_init, 'right' : self.Q_init}}) #Add state to Qtable dictionary
            action = random.choice([None, 'forward', 'left', 'right'])  #choose one of the actions at random

        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward
        if self.steps_counter > 0 :  #make sure it is not the first step in a trial.
            Q_hat = Qtable[self.state_previous][self.action_previous]
            Q_hat = Q_hat + (self.alpha * (self.reward_previous + (self.gamma * (max(Qtable[self.state].values()))) - Q_hat))
            Qtable[self.state_previous][self.action_previous] = Q_hat
            self.Qtable = Qtable

        self.state_previous = self.state
        self.action_previous = action
        self.reward_previous = reward
        self.steps_counter += 1

        #print("trip = {}, step= {},  deadline = {}, inputs = {}, action = {}, reward = {}".format(self.trips, self.steps_counter, deadline, inputs, action, reward))  # [debug]
        self.learning_table.append([self.trips, self.steps_counter, deadline, inputs, action, reward])
        
           
    def get_state(self):
        return self.state




def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=0, display=False)  # reduce update_delay to speed up simulation
    sim.run(n_trials=100)  # press Esc or close pygame window to quit
    #columns = ['tripID, teps_counter, deadline, inputs, action, reward']
    #columns=['trips', 'sucess'
    
    df = pd.DataFrame(a.learning_table)
    df.to_csv('results.csv', index=False)
    #print("Success rate: = {}".format( df.sucess.sum()* 1.0 / df.trips.count()* 1.0  ))

if __name__ == '__main__':
    run()
