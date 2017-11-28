import itertools
import numpy as np
import os
import random
import sys
import tensorflow as tf
import scipy as sp
import scipy.stats

if "./tetris/" not in sys.path:
  sys.path.append("./tetris/")

from tetris import *
from collections import deque, namedtuple

env = Tetris(use_fitness=False,action_type='oracle')

class Estimator():
    """Q-Value Estimator neural network.

    This network is used for both the Q-Network and the Target Network.
    """

    def __init__(self, scope="estimator", summaries_dir=None):
        self.scope = scope
        # Writes Tensorboard summaries to disk
        self.summary_writer = None
        with tf.variable_scope(scope):
            # Build the graph
            self._build_model()
            if summaries_dir:
                summary_dir = os.path.join(summaries_dir, "summaries_{}".format(scope))
                if not os.path.exists(summary_dir):
                    os.makedirs(summary_dir)
                self.summary_writer = tf.summary.FileWriter(summary_dir)

    def _build_model(self):
        """
        Builds the Tensorflow graph.
        """

        # Placeholders for our input
        # Our input are 4 RGB frames of shape 160, 160 each
        self.grid_states = tf.placeholder(shape=[None, 20, 10, 1], dtype=tf.uint8, name="X1")
        self.piece_states = tf.placeholder(shape=[None,45],dtype=tf.uint8,name="X2") 
        # The value of the state
        self.values = tf.placeholder(shape=[None,1], dtype=tf.float32, name="y")
        # Integer id of which action was selected
        #self.actions_pl = tf.placeholder(shape=[None,len(VALID_ACTIONS)], dtype=tf.float32, name="actions")

        X1 = tf.to_float(self.grid_states) 
        X2 = tf.to_float(self.piece_states) 
        batch_size = tf.shape(self.piece_states)[0]

        # Three convolutional layers
        conv1 = tf.contrib.layers.conv2d(
            X1, 64, 3, activation_fn=tf.nn.relu)
        conv2 = tf.contrib.layers.conv2d(
            conv1, 64, 3, activation_fn=tf.nn.relu)
        conv3 = tf.contrib.layers.conv2d(
            conv2, 128, 3, activation_fn=tf.nn.relu)

        # Fully connected layers
        fc1 = tf.contrib.layers.fully_connected(
            inputs= tf.contrib.layers.flatten(conv3), num_outputs=128)
        fc2 = tf.contrib.layers.fully_connected(
            fc1,num_outputs=256)
        fc3 = tf.contrib.layers.fully_connected(
            fc2,num_outputs=256)
        self.logits = tf.contrib.layers.fully_connected(fc3,1,activation_fn = None)

        # Get the predictions for the chosen actions only
        #gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions_pl
        #self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)

        # Calcualte the loss
        self.predictions = tf.squeeze(self.logits,squeeze_dims=[1],name='logits')
        self.losses = tf.squared_difference(self.predictions,self.values)
        self.loss = tf.reduce_sum(self.losses,name='loss')

        # Optimizer Parameters from original paper
        self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
        self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())

        # Summaries for Tensorboard
        self.summaries = tf.summary.merge([
            tf.summary.scalar("loss", self.loss),
            tf.summary.histogram("loss_hist", self.losses),
        ])


    def predict(self, sess, s1,s2):
        """
        Predicts V(s)

        Args:
          sess: Tensorflow session
          s: State input of shape [batch_size, 4, 160, 160, 3]

        Returns:
          Tensor of shape [batch_size, 1] containing the estimated 
          state values.
        """
        return sess.run(self.predictions, { self.grid_states: s1,self.piece_states: s2 })

    def update(self, sess, X1, X2, y):
        """
        Updates the estimator towards the given targets.

        Args:
          sess: Tensorflow session object
          X1: State input of shape [batch_size, 20, 10, 1]
          X2: State inpuut of shape [batch_size, 45]
          y: Targets of shape [batch_size]

        Returns:
          The calculated loss on the batch
        """
        feed_dict = { self.grid_states: X1, self.piece_states: X2, self.values: y}
        summaries, global_step, _, loss = sess.run(
            [self.summaries, tf.contrib.framework.get_global_step(), self.train_op, self.loss],
            feed_dict)
        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)
        return loss

def make_epsilon_greedy_policy(estimator,discount_factor):
    """
    Creates an epsilon-greedy policy based on all_possible_move
    Args:
        estimator: An estimator that returns q values for a given state
        nA: Number of actions in the environment.

    Returns:
        A function that takes the (sess, observation, epsilon) as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """
    def policy_fn(sess, possible_states1, possible_states2, rewards, epsilon):
        nA = len(possible_states1) 
        A = np.ones(nA, dtype=float) * epsilon / nA
        values = estimator.predict(sess, possible_states1, possible_states2)[0]
        values = values * discount_factor + np.array(rewards)
        best_action = np.argmax(values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, m-h, m+h
def test(n_times,sess,env,estimator):  
    # for testing
    '''
    n_lines = []
    n_cleared = []
    total_t = sess.run(tf.contrib.framework.get_global_step())
    n_length = []
    for i in range(100):
        state = env.reset()
        loss = None
        done = False
        total_line = 0
        total_length = 0
        total_cleared = 0
        while not done:
            action_probs = policy(sess, 0,discount_factor,env)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, line_sent, line_cleared = env.step(VALID_ACTIONS[action])
            total_line += line_sent
            total_length += 1
            total_cleared += line_cleared
            if total_length >= 200:
                break
        n_lines.append(total_line)
        n_length.append(total_length)
        n_cleared.append(total_cleared)
        print("Test " + str(i) + ": Total line cleared : " + str(total_cleared) + " Total lines sent : " + str(total_line) + " Total length:" + str(total_length))
    mean_lines = np.mean(n_lines)
    mean_length = np.mean(n_length)
    mean_cleared = np.mean(n_cleared)
    a,m,b=mean_confidence_interval(n_cleared)
    print("Mean of line cleared : " + str(m) + ' ' + str(a) + ' ' + str(b))
    print("Mean of line sent : " + str(mean_lines))
    print("Mean of game length : " + str(mean_length))
    '''
    return

EpisodeStats = namedtuple("EpisodeStats",["episode_lengths","episode_rewards","episode_line_sent","episode_line_cleared"])

def deep_q_learning(sess,
                    env,
                    estimator,
                    num_episodes,
                    experiment_dir,
                    replay_memory_size=500000,
                    replay_memory_init_size=50000,
                    update_target_estimator_every=10000,
                    discount_factor=0.99,
                    epsilon_start=1.0,
                    epsilon_end=0.1,
                    epsilon_decay_steps=2000000,
                    batch_size=32,
                    record_video_every=50):
    """
    Q-Learning algorithm for off-policy TD control using Function Approximation.
    Finds the optimal greedy policy while following an epsilon-greedy policy.

    Args:
        sess: Tensorflow Session object
        env: Tetris environment
        estimator: Estimator object used for the q values
        num_episodes: Number of episodes to run for
        experiment_dir: Directory to save Tensorflow summaries in
        replay_memory_size: Size of the replay memory
        replay_memory_init_size: Number of random experiences to sampel when initializing 
          the reply memory.
        discount_factor: Lambda time discount factor
        epsilon_start: Chance to sample a random action when taking an action.
          Epsilon is decayed over time and this is the start value
        epsilon_end: The final minimum value of epsilon after decaying is done
        epsilon_decay_steps: Number of steps to decay epsilon over
        batch_size: Size of batches to sample from the replay memory
        record_video_every: Record a video every N episodes

    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """
    #Transition = namedtuple("Transition", ["state1","state2", "action", "reward", "next_state1","next_state2", "done"])
    Transition = namedtuple("Transition", ["state1","state2", "reward", "possible_next_state1","posssible_next_state2", "done"])
    
    # The replay memory
    replay_memory = deque()

    # Keeps track of useful statistics
    stats = EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes),
        episode_line_sent=np.zeros(num_episodes),
        episode_line_cleared=np.zeros(num_episodes))
    total_sent = 0
    # Create directories for checkpoints and summaries
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "model")

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.train.Saver()
    # Load a previous checkpoint if we find one
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        print("Loading model checkpoint {}...\n".format(latest_checkpoint))
        saver.restore(sess, latest_checkpoint)

    total_t = sess.run(tf.contrib.framework.get_global_step())

    # The epsilon decay schedule
    epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

    # The policy we're following
    policy = make_epsilon_greedy_policy(
        estimator,discount_factor)
   # Populate the replay memory with initial experience
    print("Populating replay memory...")
    state = env.reset()
    for i in range(replay_memory_init_size):
        print("\r  @ Replay memory {}/{} ".format(i,replay_memory_init_size), end="")
        sys.stdout.flush()
        possible_states1,possible_states2,possible_rewards = env.get_all_possible_states()
        possible_states1,possible_states2,possible_rewards = np.array(possible_states1),np.array(possible_states2), np.array(possible_rewards)
        action_probs = policy(sess, possible_states1,possible_states2, 
                              possible_rewards, epsilons[min(total_t, epsilon_decay_steps-1)])
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        next_state, reward, done, _, _ = env.step(action)
        replay_memory.append(Transition(state[0],state[1],possible_rewards,possible_states1,possible_states2, done))
        if done:
            state = env.reset()
        else:
            state = next_state

    for i_episode in range(num_episodes):
        # Save the current checkpoint
        if i_episode % 10 == 0:
            saver.save(sess, checkpoint_path)

        # Reset the environment
        state = env.reset()
        loss = None
        total_sent = 0
        total_cleared = 0

        # One step in the environment
        for t in itertools.count():

            # Epsilon for this time step
            epsilon = epsilons[min(total_t, epsilon_decay_steps-1)]

            # Add epsilon to Tensorboard
            episode_summary = tf.Summary()
            episode_summary.value.add(simple_value=epsilon, tag="epsilon")
            estimator.summary_writer.add_summary(episode_summary, total_t)

            # Print out which step we're on, useful for debugging.
            print("\rStep {} ({}) @ Episode {}/{}, loss: {}".format(
                    t, total_t, i_episode + 1, num_episodes, loss), end="")
            sys.stdout.flush()

            # Take a step
            possible_states1,possible_states2,possible_rewards = env.get_all_possible_states()
            possible_states1,possible_states2,possible_rewards = np.array(possible_states1),np.array(possible_states2), np.array(possible_rewards)
            action_probs = policy(sess, possible_states1,possible_states2, 
                                  possible_rewards, epsilon)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, line_sent, line_cleared = env.step(action)
            

            # If our replay memory is full, pop the first element
            if len(replay_memory) == replay_memory_size:
                replay_memory.popleft()

            # Save transition to replay memory
            replay_memory.append(Transition(state[0],state[1],possible_rewards,possible_states1,possible_states2, done))

            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t
            total_sent += line_sent
            total_cleared += line_cleared
            
            # Sample a minibatch from the replay memory
            samples = random.sample(replay_memory, batch_size)
            target_batch = np.zeros((batch_size,1))
            state1_batch = []
            state2_batch = []
            for idx,sample in enumerate(samples):
                state1, state2, possible_rewards, possible_states1, possible_states2, done = sample
                # Calculate values
                values_next = estimator.predict(sess, possible_states1, possible_states2)[0]
                values = values_next * discount_factor + possible_rewards
                best_action = np.argmax(values)
                target_batch[idx] = values[best_action]
                state1_batch.append(state1)
                state2_batch.append(state2)

            # Perform gradient descent update
            states1_batch = np.array(state1_batch)
            states2_batch = np.array(state2_batch)
            loss = estimator.update(sess, states1_batch, states2_batch, target_batch)

            if done:
                break

            state = next_state
            total_t += 1

        # Add summaries to tensorboard
        episode_summary = tf.Summary()
        episode_summary.value.add(simple_value=stats.episode_rewards[i_episode], node_name="episode_reward", tag="episode_reward")
        episode_summary.value.add(simple_value=stats.episode_lengths[i_episode], node_name="episode_length", tag="episode_length")
        episode_summary.value.add(simple_value=total_sent, node_name="episode_line_sent", tag="episode_line_sent")
        episode_summary.value.add(simple_value=total_cleared, node_name="episode_line_cleared", tag="episode_line_cleared")
        estimator.summary_writer.add_summary(episode_summary, total_t)
        estimator.summary_writer.flush()

        yield total_t, EpisodeStats(
            episode_lengths=stats.episode_lengths[:i_episode+1],
            episode_rewards=stats.episode_rewards[:i_episode+1],
            episode_line_sent=stats.episode_line_sent[:i_episode+1],
            episode_line_cleared=stats.episode_line_cleared[:i_episode+1],)

    return stats


tf.reset_default_graph()

# Where we save our checkpoints and graphs
experiment_dir = os.path.abspath("./oracle_exp/{}".format('1120'))

# Create a glboal step variable
global_step = tf.Variable(0, name='global_step', trainable=False)

# Create estimators
estimator = Estimator(scope="q", summaries_dir=experiment_dir)

# State processor

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for t, stats in deep_q_learning(sess,
                                    env,
                                    estimator=estimator,
                                    experiment_dir=experiment_dir,
                                    num_episodes=100000,
                                    replay_memory_size=50000,
                                    replay_memory_init_size=5000,
                                    update_target_estimator_every=10000,
                                    epsilon_start=1.0,
                                    epsilon_end=0.1,
                                    epsilon_decay_steps=500000,
                                    discount_factor=0.99,
                                    batch_size=32):

        print("\nEpisode Length, Reward, Line Sent, Line Cleared: {},  {},  {},  {}".format(stats.episode_lengths[-1],
                                                                             stats.episode_rewards[-1],
                                                                             stats.episode_line_sent[-1],
                                                                             stats.episode_line_cleared[-1]))


