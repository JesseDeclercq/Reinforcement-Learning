from tensorflow.keras.layers import Dense,Input, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np 
import random
import pickle
import os

class SAC:

    def __init__(self):

        self.max_memory_size = 100000
        self.dim_actions = 4                                 
        self.num_frame = 4                                   
        self.dim_state = (96,96,self.num_frame)
        self.batch_size = 64
        self.replay_buffer = ReplayBuffer(self.max_memory_size, self.batch_size, self.dim_state, self.dim_actions)              
        self.gamma = 0.99
        self.tau = 0.01 # 0.005
        self.scale = 2
        self.sess = None
        self.global_step = 0
        # self.decayed_lr = tf.compat.v1.train.exponential_decay(0.001, self.global_step, 200000, 0.7, staircase=False)
        # self.optimizer_params = dict(learning_rate = self.decayed_lr, epsilon = 1e-7)
        self.learning_rate = 3e-4
        self.optimizer_params = dict(learning_rate = self.learning_rate, epsilon = 1e-7)
        self.optimizer = Adam(**(self.optimizer_params))
        self.eps = 1e-6 
        self.regularization = 1e-6

        self.actor = self.create_actor_model()
        self.critic_1 = self.create_critic_model()
        self.critic_2 = self.create_critic_model()
        self.value = self.create_value_network(trainable = True)
        self.value_target = self.create_value_network(trainable = False)

    def create_actor_model(self):
        "This function is responsible for creating the actor network."
        "The actor model recieves only the state as an input and "
        "generates a Gaussian Distribution for the action set to be sampled"
        "Therefore a mu and sigma are generated as outputs"

        wr = l2(l=self.regularization)
        state_input = Input(shape = self.dim_state)
        c1 = Conv2D(filters=8, kernel_size=(7, 7), strides=4, name='conv1', kernel_regularizer = wr, activation = 'relu')(state_input)
        mp1 = MaxPooling2D(pool_size=2, strides=2, padding='SAME')(c1)
        c2 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, name='conv2', kernel_regularizer = wr,  activation = 'relu')(mp1)
        mp2 = MaxPooling2D(pool_size=2, strides=2, padding='SAME')(c2)
        f = Flatten()(mp2)
        h1 = Dense(400, kernel_regularizer = wr, activation = 'relu')(f)
        mu = Dense(self.dim_actions, activation='softmax')(h1)             
        sigma = Dense(self.dim_actions, activation='softmax')(mu)
        actor_model = Model(inputs = state_input, outputs = [mu,sigma])
        actor_model.compile(optimizer = self.optimizer)
        return actor_model

    def create_critic_model(self):
        "This function is responsible for creating the critic networks."
        "The critic takes both the state and action as the input into the model"
        "The critic then generates a q value for the action which was taken"
        "only 1 q value is generated as we can only take one action set since this"
        "is continous control and not discrete of action choice."

        wr = l2(l=self.regularization)

        # State input section
        state_input = Input(shape = self.dim_state)
        state_c1 = Conv2D(filters=8, kernel_size=(7,7), strides=4, name='conv1', kernel_regularizer = wr, activation = 'relu')(state_input)
        state_mp1 = MaxPooling2D(pool_size=2, strides=2, padding='SAME')(state_c1)
        state_c2 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, name='conv2', kernel_regularizer = wr, activation = 'relu')(state_mp1)
        state_mp2 = MaxPooling2D(pool_size=2, strides=2, padding='SAME')(state_c2)
        state_f = Flatten()(state_mp2)

        # The action input section
        action_input = Input(shape = self.dim_actions)
        action_h1 = Dense(400, activation = 'relu')(action_input)

        # combining the action and state inputs
        merged = Concatenate()([action_h1, state_f])
        merged_h1 = Dense(400, activation = 'relu', kernel_regularizer=wr)(merged)
        output = Dense(1, activation = None, kernel_regularizer = wr)(merged_h1)
        critic_model = Model(inputs = [state_input, action_input], outputs = output)
        critic_model.compile(optimizer = self.optimizer)
        return critic_model

    def create_value_network(self, trainable):
        "The value network is required for the soft update which in turn is the disciminator for the critic network."
        "The value network takes only the state as an input and returns the V value for the Q critic update."

        if trainable:
            wr = l2(l=self.regularization)
        else:
            wr = None

        # State input section
        input = Input(shape = self.dim_state)
        c1 = Conv2D(filters=8, kernel_size=(7,7), strides=4, name='conv1', kernel_regularizer = wr, activation = 'relu')(input)
        mp1 = MaxPooling2D(pool_size=2, strides=2, padding='SAME')(c1)
        c2 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, name='conv2', kernel_regularizer = wr, activation = 'relu')(mp1)
        mp2 = MaxPooling2D(pool_size=2, strides=2, padding='SAME')(c2)
        f = Flatten()(mp2)
        h1 = Dense(400, activation = 'relu', kernel_regularizer=wr)(f)
        output = Dense(1, activation = None, kernel_regularizer = wr)(h1)
        value_model = Model(inputs = input, outputs = output)
        value_model.compile(optimizer = self.optimizer)
        return value_model

    def select_action(self, state):
        "This function is responsible for choosing the appropriate action according to the policy (actor)"
        "as well as abdiding to the epsilon-greedy policy."
        
        state = tf.convert_to_tensor(state)
        action, log_probs = self.actor_sample(state, reparameterize=False)

        return action

    def actor_sample(self, state, reparameterize=True):
        "This function samples from the actor's output"

        mu, sigma = self.actor(state)
        mu = abs(mu) + self.eps; 
        sigma = abs(sigma) + self.eps                                                                    
        probabilities = tfp.distributions.Normal(loc = mu, scale = sigma)      # probabilities for the actions taken 

        # Here we sample random actions.
        if reparameterize:
            actions = probabilities.sample() # + something else if you want to implement
        else:
            actions = probabilities.sample()

        action = tf.math.softmax(actions + self.eps) 
        log_probs = probabilities.log_prob(actions + self.eps)
        log_probs = tf.math.reduce_sum(log_probs, axis=1, keepdims=True)

        return action, log_probs

    def update_network_parameters(self, tau = None):
        "This function is responsible for updating"

        if tau is None: tau = self.tau
        weights = []
        targets = self.value_target.weights
        for i, weight in enumerate(self.value.weights):
            weights.append(weight * tau + targets[i]*(1-tau))

        self.value_target.set_weights(weights)

    def train(self):
        "This is the training function of the SAC algorithm"

        if len(self.replay_buffer.memory) < self.batch_size:
            return
   
        state, action, reward, new_state, done = self.replay_buffer.sample_memories()
        states = tf.convert_to_tensor(state, dtype=tf.float32)
        next_states = tf.convert_to_tensor(new_state, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)

        #================================Updating the value network==============================================
        with tf.GradientTape() as tape:
            value = tf.squeeze(self.value(states))                
            next_value = tf.squeeze(self.value_target(next_states))     
            current_policy_actions, log_probs = self.actor_sample(states, reparameterize=False)
            log_probs = tf.squeeze(log_probs)
            q1_new_policy = self.critic_1([states, current_policy_actions])
            q2_new_policy = self.critic_2([states, current_policy_actions])
            critic_value = tf.squeeze(tf.math.minimum(q1_new_policy, q2_new_policy)) # we choose the minimum value between the two critics as the Q value from the critic
            value_target = critic_value - log_probs
            value_loss = 0.5 * tf.keras.losses.MSE(value, value_target)

        value_network_gradient = tape.gradient(value_loss,self.value.trainable_variables)
        self.value.optimizer.apply_gradients(zip(value_network_gradient, self.value.trainable_variables))

        #================================Updating the actor network==============================================
        with tf.GradientTape() as tape:
            new_policy_actions, log_probs = self.actor_sample(states,reparameterize=True) # in the original paper, they reparameterize here. We dont so this so it's just the usual action.
            log_probs = tf.squeeze(log_probs)
            q1_new_policy = self.critic_1([states, new_policy_actions])
            q2_new_policy = self.critic_2([states, new_policy_actions])
            critic_value = tf.squeeze(tf.math.minimum(q1_new_policy, q2_new_policy), 1) # we choose the minimum value between the two critics as the Q value from the critic
            actor_loss = log_probs - critic_value
            actor_loss = tf.math.reduce_mean(actor_loss)

        actor_network_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_network_gradient, self.actor.trainable_variables))
        
        #================================Updating the critic networks==============================================
        with tf.GradientTape(persistent=True) as tape:
            q_hat = self.scale*reward + self.gamma*next_value*(1-done) # I didn't know that these context managers shared values?
            q1_old_policy = tf.squeeze(self.critic_1([state, action]))
            q2_old_policy = tf.squeeze(self.critic_2([state, action]))
            critic_1_loss = 0.5 * tf.keras.losses.MSE(q1_old_policy, q_hat)
            critic_2_loss = 0.5 * tf.keras.losses.MSE(q2_old_policy, q_hat)
    
        critic_1_network_gradient = tape.gradient(critic_1_loss,self.critic_1.trainable_variables)
        critic_2_network_gradient = tape.gradient(critic_2_loss,self.critic_2.trainable_variables)

        self.critic_1.optimizer.apply_gradients(zip(critic_1_network_gradient, self.critic_1.trainable_variables))
        self.critic_2.optimizer.apply_gradients(zip(critic_2_network_gradient, self.critic_2.trainable_variables))

        self.update_network_parameters()
        self.global_step += 1

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
            ix_to_remove = random.randint(0,self.max_memory_size-1)
            self.memory.pop(ix_to_remove) 
        self.memory.append({'state':state, 'action':action, 'reward':reward, 'next_state':next_state, 'done': done})

    def sample_memories(self):
        "Here we sample from the memory"

        samples = random.sample(self.memory, self.batch_size)
        states = np.zeros((self.batch_size, self.state_shape[0], self.state_shape[1], self.state_shape[2]))
        actions = np.zeros((self.batch_size, self.action_shape))
        next_states = np.zeros((self.batch_size, self.state_shape[0], self.state_shape[1], self.state_shape[2]))
        rewards = np.zeros(self.batch_size)
        done = np.zeros(self.batch_size)

        for i,sample in enumerate(samples):
            states[i] = sample['state']
            actions[i] = sample['action']
            rewards[i] = sample['reward']
            next_states[i] = sample['next_state']
            done[i] = sample['done']*1 # to get to integer

        return states, actions, rewards, next_states, done

class Agent(SAC):
    def __init__(self, env):
        super(SAC, self).__init__()

        self.env = env
        self.sess = None
        self.agent = SAC()
        self.checkpoint_freq = 50 
        self.state = np.zeros((1,self.agent.dim_state[0], self.agent.dim_state[1], self.agent.dim_state[2]))
        self.reward_check = 50
        self.reward_memory_size = 100000
        self.agent.actor.checkpoint_file = 'Checkpoints/Actor/actor_sac'
        self.agent.critic_1.checkpoint_file = 'Checkpoints/Critic1/critic1_sac'
        self.agent.critic_2.checkpoint_file = 'Checkpoints/Critic2/critic2_sac'
        self.agent.value.checkpoint_file = 'Checkpoints/Value/valuer_sac'
        self.agent.value_target.checkpoint_file = 'Checkpoints/Value_Target/value_target_sac'
        self.rewards_checkpoint_file = 'Checkpoints/Rewards/'
        self.buffer_checkpoint_file = 'Checkpoints/Replay/'
        
    def _convert_action(self, action):
        "convert the 4th order action into a 3rd order for the environment step"
        "Adding the two steering components together. the left component is negative however"
        action = tf.squeeze(action)
        action_3 = np.zeros(3)
        action_3[0] = action[1] - action[0] 
        action_3[1:] = action[2:]
        return action_3

    def _update_state(self, state, next_state):
        "This updates the existing state with the next state"
        "3 is newest 0 is oldest"

        for s in range(state.shape[-1] - 1): state[:,:,:,s] = state[:,:,:,s+1]
        state[:,:,:,-1] = next_state
        return state

    def _reward(self, reward):
        "Here we place our own internal check to make sure the car doesn take too many consecutive negative rewards"
        
        if len(self.rewards) % self.reward_memory_size == 0 and self.rewards : 
            self.rewards.pop(0)
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
            if episode < 200 and episode_step < 200:
                action = np.array([0,0,1,0]) # accelerate only for the first start
            else:
                action = self.agent.select_action(self.state)
            env_action = self._convert_action(action)
            episode_next_state, reward, done, info = self.env.step(env_action) 
            done = self._reward(reward)
            next_state = self._update_state(self.state, episode_next_state)
            if train:
                self.agent.make_memory(self.state, action, reward, next_state, done)
                self.agent.train()
            self.state = next_state
        else: done = False; reward = 0
            
        # =====================================Saving Model & Buffer=====================================
        if episode % self.checkpoint_freq == 0 and episode_step == 1 and train: 
            self.save_models()
            self.save_buffer()

        return done, reward
            
    def save_models(self):
        print('... saving models ...')
        self.agent.actor.save_weights(self.agent.actor.checkpoint_file)
        self.agent.critic_1.save_weights(self.agent.critic_1.checkpoint_file)
        self.agent.critic_2.save_weights(self.agent.critic_2.checkpoint_file)
        self.agent.value.save_weights(self.agent.value.checkpoint_file)
        self.agent.value_target.save_weights(self.agent.value_target.checkpoint_file)

    def save_rewards(self, rewards, name):
        reward_checkpoint_file = os.path.join(self.rewards_checkpoint_file, name+'.txt')
        rewards_file = open(reward_checkpoint_file, 'w')
        for row in np.array(rewards).reshape(len(rewards),1):
            np.savetxt(rewards_file, row)
        rewards_file.close()

    def save_buffer(self):
        print('... saving buffer ...')
        buffer_checkpoint_file = os.path.join(self.buffer_checkpoint_file, 'Buffer.p')
        pickle.dump(self.agent.replay_buffer.memory, open(buffer_checkpoint_file,'wb'))

    def load_models(self):
        print('... loading models ...')
        self.agent.actor.load_weights(self.agent.actor.checkpoint_file)
        self.agent.critic_1.load_weights(self.agent.critic_1.checkpoint_file)
        self.agent.critic_2.load_weights(self.agent.critic_2.checkpoint_file)
        self.agent.value.load_weights(self.agent.value.checkpoint_file)
        self.agent.value_target.load_weights(self.agent.value_target.checkpoint_file)

    def load_rewards(self, name):
        print('... loading rewards ...')
        rewards_file = os.path.join(self.rewards_checkpoint_file, name + '.txt')
        rewards = np.loadtxt(rewards_file)
        return rewards

    def load_buffer(self):
        print("...loading buffer...")
        buffer_checkpoint_file = os.path.join(self.buffer_checkpoint_file, 'Buffer.p')
        memory = pickle.load(open(buffer_checkpoint_file, "rb") )
        self.agent.replay_buffer.memory = memory

