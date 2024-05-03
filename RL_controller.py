
import numpy as np

#CONTROLLER STABLIZES ON 8TH TRIAL

class RL_controller:
    def __init__(self, args):
        self.gamma = args.gamma
        self.lr = args.lr #scales update to Q values
        self.Q_value = np.zeros((args.theta_discrete_steps, args.theta_dot_discrete_steps, 3)) # state-action values
        self.V_values = np.zeros((args.theta_discrete_steps, args.theta_dot_discrete_steps)) # state values
        self.prev_a = 0 # previous action
        # Use a previous_state = None to detect the beginning of the new round e.g. if not(self.prev_s is None): ...
        self.prev_s = None # Previous state

    def reset(self):
        #You need to reset sth
        #self.prev_a = 0
        self.prev_s = None #reset these values to indicate the start of a new round
        self.prev_a = 0
        print("Controller reset")

    def get_action(self, state, image_state, random_controller=False, episode=0):
        terminal, timestep, theta, theta_dot, reward = state
        
        action = np.random.rand()
        if random_controller or action > 0.8:
            action = np.random.randint(0, 3)
            
        else:
            theta_index = int(theta)
            theta_dot_index = int(theta_dot) #gets index of theta for Q array
            
            action = np.argmax(self.Q_value[theta_index, theta_dot_index, :]) #max action across all actions given this theta and theta dot
            
            if not (self.prev_s is None or self.prev_s == [theta, theta_dot]):
                #calculate Q values
                prev_theta_index = int(self.prev_s[0]) #locate previous Q value
                prev_theta_dot_index = int(self.prev_s[1])
                prev_action = self.prev_a
                
                #updates Q Value based on Bellman equation: updates previous state/action pair based on the difference
                # between current reward and estimated future eward
                # scaled by learning factor with reward and discount factor
                self.Q_value[prev_theta_index, prev_theta_dot_index, prev_action] += self.lr * (
                    reward + self.gamma * np.max(
                        self.Q_value[theta_index, theta_dot_index, :]) - #colon shorthand for all possible actions given the current theta and theta dot
                    self.Q_value[prev_theta_index, prev_theta_dot_index, prev_action])
                
                self.V_values[theta_index, theta_dot_index] = max(self.Q_value[theta_index, theta_dot_index, :])
                
        self.prev_s = [theta, theta_dot]
        self.prev_a = action
        return action

