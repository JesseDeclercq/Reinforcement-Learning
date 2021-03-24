from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import numpy as np 
import random
import os

class DDPG:
    "This is the class which defines the deep deterministic policy gradient model."
    "The total number of actions is 4 : left right forward brake"

    def __init__(self, env, sess):
        

        self.env = env
        self.sess = sess
        self.max_memory_size = 100000
        self.dim_actions = 4                                 
        self.num_frame = 4                                   
        self.dim_state = (96,96,self.num_frame)
        self.batch_size = 32
        self.replay_buffer = ReplayBuffer(self.max_memory_size, self.batch_size, self.dim_state, self.dim_actions)            
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.gamma = 0.95
        self.tau = 0.125
        self.regularization = 1e-6
        self.global_step = 0
        self.decayed_lr = tf.compat.v1.train.exponential_decay(0.001, self.global_step, 200000, 0.7, staircase=False)
        self.optimizer_params = dict(learning_rate = self.decayed_lr, epsilon = 1e-7)
        self.optimizer = tf.compat.v1.train.AdamOptimizer(**(self.optimizer_params))
        self.build_graph()   
        
    
    def build_graph(self):
        "This function builds a tensorflow graph or essentially explains the mathematical processes which need to take place"
        "This enables the tensorflow session to communicate with the variables as well as gradients"

        #===========================Actor=============================== de/dA as = de/dC * dC/dA
        self.actor = self.create_actor_model(trainable = True)
        self.actor_state_input = self.actor.input
        self.actor_target = self.create_actor_model(trainable = False) 
        self.actor_critic_grad = tf.compat.v1.placeholder(tf.float32, [None, self.dim_actions])                # where we will feed de/dC (from critic)
        self.actor_grads = tf.gradients(self.actor.output, self.actor.trainable_weights, -self.actor_critic_grad) # Dont understand
        grads = zip(self.actor_grads, self.actor.trainable_weights)
        self.actor_optimizer = tf.compat.v1.train.AdamOptimizer(**(self.optimizer_params)).apply_gradients(grads)     # else just lr = 0.001

        #========================Critic==================================
        self.critic = self.create_critic_model(trainable = True)              
        self.critic_state_input = self.critic.inputs[0]
        self.critic_action_input = self.critic.inputs[1]
        self.critic_target = self.create_critic_model(trainable = False)           
        self.critic_grads = tf.gradients(self.critic.output, self.critic_action_input)

        
        # Initialize for later gradient calculations
        self.sess.run(tf.compat.v1.global_variables_initializer())

    def create_actor_model(self, trainable):
        "This function is responsible for creating the actor and the critic networks."
        "The actor and critic networks have the same baseline architecture, however,"
        "The actor critic takes only the state as an input and returns a "
        "softmax for acceleration and brake and a tanh for steering -1 = left, +1 = right"
        "The critic network takes in the output of the actor as well as the next state as inputs"
        "And returns Q values for each of the states, therefore linear activation functions."
        "Both the actor and critic networks require target networks which are updated at a set frequency."

        if trainable:
            wr = l2(l=self.regularization)
        else:
            wr = None

        state_input = Input(shape = self.dim_state)
        c1 = Conv2D(filters=8, kernel_size=(7,7), strides=4, name='conv1', kernel_regularizer = wr, activation = 'relu')(state_input)
        mp1 = MaxPooling2D(pool_size=2, strides=2, padding='SAME')(c1)
        c2 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, name='conv2', kernel_regularizer = wr, activation = 'relu')(mp1)
        mp2 = MaxPooling2D(pool_size=2, strides=2, padding='SAME')(c2)
        f = Flatten()(mp2)
        h1 = Dense(400, activation = 'relu', kernel_regularizer = wr)(f)
        output = Dense(self.dim_actions, activation = 'softmax', kernel_regularizer = wr)(h1)
        actor_model = Model(inputs = state_input, outputs = output)
        actor_model.compile(loss = 'mse', optimizer = self.optimizer)
         
        return actor_model

    def create_critic_model(self,trainable):
        "This function is responsible for creating the actor and the critic networks."
        "The actor and critic networks have the same baseline architecture, however,"
        "The actor critic takes only the state as an input and returns a "
        "softmax for acceleration and brake and a tanh for steering -1 = left, +1 = right"
        "The critic network takes in the output of the actor as well as the next state as inputs"
        "And returns Q values for each of the states, therefore linear activation functions."
        "Both the actor and critic networks require target networks which are updated at a set frequency."

        if trainable:
            wr = l2(l=self.regularization)
        else:
            wr = None

        # State input section
        state_input = Input(shape = self.dim_state)
        state_c1 = Conv2D(filters=8, kernel_size=(7,7), strides=4, name='conv1', kernel_regularizer = wr, activation = 'relu')(state_input)
        state_mp1 = MaxPooling2D(pool_size=2, strides=2, padding='SAME')(state_c1)
        state_c2 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, name='conv2', kernel_regularizer = wr, activation = 'relu')(state_mp1)
        state_mp2 = MaxPooling2D(pool_size=2, strides=2, padding='SAME')(state_c2)
        state_f = Flatten()(state_mp2)

        # The action input section
        action_input = Input(shape = self.dim_actions)
        action_h1 = Dense(64, activation = 'relu')(action_input)

        # combining the action and state inputs
        merged = Concatenate()([action_h1, state_f])
        merged_h1 = Dense(400, activation = 'relu', kernel_regularizer=wr)(merged)
        output = Dense(1, activation = None, kernel_regularizer = wr)(merged_h1)
        critic_model = Model(inputs = [state_input, action_input], outputs = output)
        critic_model.compile(loss = 'mse', optimizer = self.optimizer)

        return critic_model

    def _train_actor(self, states, next_states, rewards, actions, not_done):
        "This function trains the actor network using random samples of a batch size sampled randomly from the experience buffer"

        predicted_actions = self.actor.predict(states)
        grads = self.sess.run(self.critic_grads, feed_dict = {self.critic_state_input: states, self.critic_action_input: predicted_actions})[0]
        self.sess.run(self.actor_optimizer, feed_dict = {self.actor_state_input: states, self.actor_critic_grad: grads}) 

    def _train_critic(self, states, next_states, rewards, actions, not_done): 
        "This function trains the critic using various samples from a designated batch size randomly sampled from the experience buffer."

        target_action = self.actor_target.predict(next_states)
        future_reward = self.critic_target.predict([next_states, target_action])
        q_target = future_reward*self.gamma*not_done + rewards

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)

        self.critic.fit([states, actions], q_target, batch_size = 64, epochs = 1, steps_per_epoch = 1, verbose = 0)

    def train(self):
        "This function is called to train the critic and actor networks according to their respective loss functions"
        "The actor needs to be updated using the chain rule according to the loss of the Bellman Equation for the Critic Network"
        "As well as the policy by which the actor is parameterized."

        if len(self.replay_buffer.memory) < self.batch_size:
            return

        state, action, reward, next_state, done = self.replay_buffer.sample_memories()
        not_done = tf.cast(tf.logical_not(tf.cast(done, "bool")), "float32") # this ensures that if done is True we mutliply by a 0

        self._train_actor(state, next_state, reward, action, not_done)
        self._train_critic(state, next_state, reward, action, not_done)
        self.global_step += 1

    def _update_actor_target(self):
        "This function updates the weights of the actor target network."
        "This function seems innefficient and can be changed"

        actor_model_weights = self.actor.get_weights()
        actor_target_weights = self.critic_target.get_weights()
        for i in range(len(actor_target_weights)):
            actor_target_weights[i] = actor_model_weights[i]
        self.critic_target.set_weights(actor_target_weights)

    def _update_critic_target(self):
        "This function updates the weights of the actor target network."
        "This function seems innefficient and can be changed"

        critic_model_weights  = self.critic.get_weights()
        critic_target_weights = self.critic_target.get_weights()

        for i in range(len(critic_target_weights)):
            critic_target_weights[i] = critic_model_weights[i]
        self.critic_target.set_weights(critic_target_weights)		

    def update_targets(self):
        "This function executes the updating of the target networks"

        self._update_actor_target()
        self._update_critic_target()
 
    def choose_action(self, current_state):
        "This function chooses an action based on the epsilon-greedy model"

        self.epsilon *= self.epsilon_decay
        if np.random.random() < self.epsilon:
            action = self.env.action_space.sample()
            model_action = self.convert_action(action, elements = 4)
        else:
            model_action = self.actor.predict(current_state)[0]
            action = self.convert_action(model_action, elements = 3)
        return model_action, action

    def convert_action(self, action, elements):
        "action_4 = [left,right,acc,brake]"
        "action_3 = [-1:1,0:1,0:1]"

        if elements == 3:
            action_3 = np.zeros(3)
            action_3[0] = action[1] - action[0] # adding the two steering components together. the left component is negative however
            action_3[1:] = action[2:]
            return action_3
        elif elements == 4:
            action_4 = np.zeros(4)
            if action[0] < 0:
                action_4[0] = -action[0]
            else:
                action_4[1] = action[0] 
            action_4[2:] = action[1:]
            return action_4
    
    def make_memory(self, state, action, reward, next_state, done):
        "Here we remember what has happened so sampling can occur"

        self.replay_buffer.store_memory(state, action, reward, next_state, done)

class ReplayBuffer:

    def __init__(self, max_memory_size, batch_size, state_shape, action_shape):
        self.memory = []
        self.batch_size = batch_size
        self.max_memory_size = max_memory_size
        self.state_shape = state_shape
        self.action_shape = action_shape

    def store_memory(self, state, action, reward, next_state, done):
        "We add a memory to the replay buffer and make sure it doesnt go past "
        "We opt to remove random experiences rather than ones which are necessarily the oldest"

        if len(self.memory) >= self.max_memory_size:
            ix_to_remove = random.randint(0,self.max_memory_size)
            self.memory.pop(ix_to_remove) 
        self.memory.append({'state':state, 'action':action, 'reward':reward, 'next_state':next_state, 'done': done})

    def sample_memories(self):
        "Here we sample from the memory"

        samples = random.sample(self.memory, self.batch_size)
        states = np.zeros((self.batch_size, self.state_shape[0], self.state_shape[1], self.state_shape[2]))
        actions = np.zeros((self.batch_size, self.action_shape))
        next_states = np.zeros((self.batch_size, self.state_shape[0], self.state_shape[1], self.state_shape[2]))
        rewards = np.zeros(self.batch_size)
        done = np.zeros(self.batch_size, dtype = 'bool')

        for i,sample in enumerate(samples):
            states[i] = sample['state']
            actions[i] = sample['action']
            rewards[i] = sample['reward']
            next_states[i] = sample['next_state']
            done[i] = sample['done'] # we need to keep as boolean so the tf.logical_not can work

        return states, actions, rewards, next_states, done

class Agent(DDPG):
    
    def __init__(self, env, sess):
        self.env = env
        self.sess = sess
        self.agent = DDPG(self.env, self.sess)
        self.checkpoint_freq = 100 
        self.state = np.zeros((1,self.agent.dim_state[0], self.agent.dim_state[1], self.agent.dim_state[2]))
        self.reward_check = 50
        self.reward_memory_size = 1000
        self.agent.actor.checkpoint_file = 'checkpoints/actor/actor_ddpg'
        self.agent.actor_target.checkpoint_file = 'checkpoints/actor_target/actor_target_ddpg'
        self.agent.critic.checkpoint_file = 'checkpoints/critic/critic_ddpg'
        self.agent.critic_target.checkpoint_file = 'checkpoints/critic_target/critic_target_ddpg'
        self.rewards_checkpoint_file = 'checkpoints/rewards/'

    def _update_state(self, state, next_state):
        "This updates the existing state with the next state"
        "3 is newest 0 is oldest"

        for s in range(state.shape[-1] - 1): state[:,:,:,s] = state[:,:,:,s+1]
        state[:,:,:,-1] = next_state
        return state

    def _reward(self, reward):
        "Here we place our own internal check to make sure the car doesn take too many consecutive negative rewards"
        
        if len(self.rewards) % self.reward_memory_size == 0 and self.rewards : self.rewards.pop(0)
        self.rewards.append(reward)
        if len(self.rewards) > self.reward_check:
            done = True
            r = 1
            while r < self.reward_check:
                if self.rewards[-r] > 0:
                    done = False
                    break
                r += 1
        else:
            done = False

        return done
         
    def step(self, train, episode, episode_step):
        "This function is responsible for the main operation of the Agent"

        single_state = self.env.get_state()
        self.state[:,:,:,episode_step % self.agent.num_frame - 1] = single_state
        if episode_step == 1: self.rewards = []
        if episode_step % self.agent.num_frame == 0:
            #==========Providing artifical start to the episode:==========
            if episode < 50 and episode_step < 200:
                action = np.array([0,0,1,0]) # accelerate only for the first start
                env_action = np.array([0,1,0])
            else:
                action, env_action = self.agent.choose_action(self.state)

            episode_next_state, reward, done, info = self.env.step(env_action) 
            done = self._reward(reward)
            next_state = self._update_state(self.state, episode_next_state)

            #========================training the model============================
            if train:
                self.agent.make_memory(self.state, action, reward, next_state, done)
                self.agent.train()
            self.state = next_state
        else: done = False; reward = 0
            
        # Saving Model
        # if episode % self.checkpoint_freq == 0 and episode_step == 1 and train: self.save_models()

        return done, reward
            
    def save_models(self):
        print('... saving models ...')
        self.agent.actor.save_weights(self.agent.actor.checkpoint_file)
        self.agent.actor_target.save_weights(self.agent.actor_target.checkpoint_file)
        self.agent.critic.save_weights(self.agent.critic.checkpoint_file)
        self.agent.critic_target.save_weights(self.agent.critic_target.checkpoint_file)
        
    def load_models(self):
        print('... loading models ...')
        self.agent.actor.load_weights(self.agent.actor.checkpoint_file)
        self.agent.actor_target.load_weights(self.agent.actor_target.checkpoint_file)
        self.agent.critic.load_weights(self.agent.critic.checkpoint_file)
        self.agent.critic_target.load_weights(self.agent.critic_target.checkpoint_file)
        self.rewards = np.loadtxt(self.rewards_checkpoint_file)

    def save_rewards(self, rewards, name):
        reward_checpoint_file = os.path.join(self.rewards_checkpoint_file, name+'.txt')
        rewards_file = open(reward_checpoint_file, 'w')
        for row in np.array(rewards).reshape(len(rewards),1):
            np.savetxt(rewards_file, row)
        rewards_file.close()