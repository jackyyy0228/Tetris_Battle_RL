import itertools
import numpy as np
import os
import random
import sys
import tensorflow as tf
import scipy as sp
import scipy.stats

if "../../" not in sys.path:
  sys.path.append("../../")

if "../PolicyGradient/a3c/" not in sys.path:
  sys.path.append("../PolicyGradient/a3c/")

from tetris import *
from lib import plotting
from collections import deque, namedtuple

#env = gym.envs.make("Breakout-v0")
env = Tetris(use_fitness=True,action_type='grouped')
# Atari Actions: 0 (noop), 1 (fire), 2 (left) and 3 (right) are valid actions
VALID_ACTIONS = list(range(53))


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
        self.piece_states = tf.placeholder(shape=[None,48],dtype=tf.uint8,name="X2") 
        # The TD target value
        self.y_pl = tf.placeholder(shape=[None,len(VALID_ACTIONS)], dtype=tf.float32, name="y")
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
        self.predictions = tf.contrib.layers.fully_connected(fc3, len(VALID_ACTIONS))

        # Get the predictions for the chosen actions only
        #gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions_pl
        #self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)

        # Calcualte the loss
        #self.losses = tf.squared_difference(self.y_pl, self.action_predictions)
        #self.loss = tf.reduce_mean(self.losses)
        self.losses = tf.squared_difference(tf.reshape(self.y_pl,[-1]),tf.reshape(self.predictions,[-1]) )
        self.loss = tf.reduce_mean(self.losses)

        # Optimizer Parameters from original paper
        self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
        self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())

        # Summaries for Tensorboard
        self.summaries = tf.summary.merge([
            tf.summary.scalar("loss", self.loss),
            tf.summary.histogram("loss_hist", self.losses),
            tf.summary.histogram("q_values_hist", self.predictions),
            tf.summary.scalar("max_q_value", tf.reduce_max(self.predictions))
        ])


    def predict(self, sess, s1,s2):
        """
        Predicts action values.

        Args:
          sess: Tensorflow session
          s: State input of shape [batch_size, 4, 160, 160, 3]

        Returns:
          Tensor of shape [batch_size, NUM_VALID_ACTIONS] containing the estimated 
          action values.
        """
        return sess.run(self.predictions, { self.grid_states: s1,self.piece_states: s2 })

    def update(self, sess, X1, X2, y):
        """
        Updates the estimator towards the given targets.

        Args:
          sess: Tensorflow session object
          s: State input of shape [batch_size, 4, 160, 160, 3]
          a: Chosen actions of shape [batch_size]
          y: Targets of shape [batch_size]

        Returns:
          The calculated loss on the batch
        """
        feed_dict = { self.grid_states: X1, self.piece_states: X2, self.y_pl: y}
        summaries, global_step, _, loss = sess.run(
            [self.summaries, tf.contrib.framework.get_global_step(), self.train_op, self.loss],
            feed_dict)
        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)
        return loss

def copy_model_parameters(sess, estimator1, estimator2):
    """
    Copies the model parameters of one estimator to another.

    Args:
      sess: Tensorflow session instance
      estimator1: Estimator to copy the paramters from
      estimator2: Estimator to copy the parameters to
    """
    e1_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator1.scope)]
    e1_params = sorted(e1_params, key=lambda v: v.name)
    e2_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator2.scope)]
    e2_params = sorted(e2_params, key=lambda v: v.name)

    update_ops = []
    for e1_v, e2_v in zip(e1_params, e2_params):
        op = e2_v.assign(e1_v)
        update_ops.append(op)

    sess.run(update_ops)


def make_epsilon_greedy_policy(estimator, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.

    Args:
        estimator: An estimator that returns q values for a given state
        nA: Number of actions in the environment.

    Returns:
        A function that takes the (sess, observation, epsilon) as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """
    def policy_fn(sess, observation, epsilon):
        A = np.ones(nA, dtype=float) * epsilon / nA
        #q_values = estimator.predict(sess, np.expand_dims(observation, 0))[0]
        q_values = estimator.predict(sess, np.expand_dims(observation[0],0),
                                     np.expand_dims(observation[1],0) )[0]
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, m-h, m+h
def deep_q_learning(sess,
                    env,
                    q_estimator,
                    target_estimator,
                    num_episodes,
                    experiment_dir,
                    replay_memory_size=500000,
                    replay_memory_init_size=50000,
                    update_target_estimator_every=10000,
                    discount_factor=0.99,
                    epsilon_start=1.0,
                    epsilon_end=0.1,
                    epsilon_decay_steps=500000,
                    batch_size=32,
                    record_video_every=50):
    """
    Q-Learning algorithm for fff-policy TD control using Function Approximation.
    Finds the optimal greedy policy while following an epsilon-greedy policy.

    Args:
        sess: Tensorflow Session object
        env: OpenAI environment
        q_estimator: Estimator object used for the q values
        target_estimator: Estimator object used for the targets
        state_processor: A StateProcessor object
        num_episodes: Number of episodes to run for
        experiment_dir: Directory to save Tensorflow summaries in
        replay_memory_size: Size of the replay memory
        replay_memory_init_size: Number of random experiences to sampel when initializing 
          the reply memory.
        update_target_estimator_every: Copy parameters from the Q estimator to the 
          target estimator every N steps
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
    Transition = namedtuple("Transition", ["state1","state2", "action", "reward", "next_state1","next_state2", "done"])

    # The replay memory
    replay_memory = []

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))
    total_sent = 0
    # Create directories for checkpoints and summaries
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "model")
    monitor_path = os.path.join(experiment_dir, "monitor")

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(monitor_path):
        os.makedirs(monitor_path)

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
        q_estimator,
        len(VALID_ACTIONS))
    # for testing
    n_lines = []
    n_cleared = []
    n_length = []
    for i in range(100):
        state = env.reset()
        loss = None
        done = False
        total_line = 0
        total_length = 0
        total_cleared = 0
        while not done:
            action_probs = policy(sess, state, 0)
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

    exit()
    # Populate the replay memory with initial experience
    print("Populating replay memory...")
    state = env.reset()
    for i in range(replay_memory_init_size):
        action_probs = policy(sess, state, epsilons[min(total_t, epsilon_decay_steps-1)])
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        next_state, reward, done, _, _ = env.step(VALID_ACTIONS[action])
        replay_memory.append(Transition(state[0],state[1], action, reward, next_state[0],next_state[1], done))
        if done:
            state = env.reset()
        else:
            state = next_state
    # Record videos
    # Use the gym env Monitor wrapper

    for i_episode in range(num_episodes):

        # Save the current checkpoint
        saver.save(tf.get_default_session(), checkpoint_path)

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
            q_estimator.summary_writer.add_summary(episode_summary, total_t)

            # Maybe update the target estimator
            if total_t % update_target_estimator_every == 0:
                copy_model_parameters(sess, q_estimator, target_estimator)
                print("\nCopied model parameters to target network.")

            # Print out which step we're on, useful for debugging.
            print("\rStep {} ({}) @ Episode {}/{}, loss: {}".format(
                    t, total_t, i_episode + 1, num_episodes, loss), end="")
            sys.stdout.flush()

            # Take a step
            action_probs = policy(sess, state, epsilon)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, line_sent, line_cleared = env.step(VALID_ACTIONS[action])
            #next_state = np.append(state[:,:,1:], np.expand_dims(next_state, 2), axis=2)

            # If our replay memory is full, pop the first element
            if len(replay_memory) == replay_memory_size:
                replay_memory.pop(0)

            # Save transition to replay memory
            replay_memory.append(Transition(state[0],state[1],action, reward, next_state[0],next_state[1],done))   

            # Update statistics
            stats.episode_rewards[i_episode] += reward[action]
            stats.episode_lengths[i_episode] = t
            total_sent += line_sent
            total_cleared += line_cleared
            
            # Sample a minibatch from the replay memory
            samples = random.sample(replay_memory, batch_size)
            states1_batch,states2_batch, action_batch, reward_batch, next_states1_batch, next_states2_batch,done_batch = map(np.array, zip(*samples))

            # Calculate q values and targets (Double DQN)
            q_values_next = q_estimator.predict(sess, next_states1_batch,next_states2_batch)
            best_actions = np.argmax(q_values_next, axis=1)
            q_values_next_target = target_estimator.predict(sess, next_states1_batch,next_states2_batch)
            #targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * \
            #    discount_factor * q_values_next_target[np.arange(batch_size), best_actions]

            # Perform gradient descent update
            states1_batch = np.array(states1_batch)
            states2_batch = np.array(states2_batch)
            #reward_batch = np.array(reward_batch)
            reward_batch = np.vstack(reward_batch)
            loss = q_estimator.update(sess, states1_batch,states2_batch, reward_batch)

            if done:
                break

            state = next_state
            total_t += 1

        # Add summaries to tensorboard
        episode_summary = tf.Summary()
        episode_summary.value.add(simple_value=stats.episode_rewards[i_episode], node_name="episode_reward", tag="episode_reward")
        episode_summary.value.add(simple_value=stats.episode_lengths[i_episode], node_name="episode_length", tag="episode_length")
        episode_summary.value.add(simple_value=total_sent, node_name="episode_line_sent", tag="episode_line_sent")
        q_estimator.summary_writer.add_summary(episode_summary, total_t)
        q_estimator.summary_writer.flush()

        yield total_t, plotting.EpisodeStats(
            episode_lengths=stats.episode_lengths[:i_episode+1],
            episode_rewards=stats.episode_rewards[:i_episode+1])

    return stats


tf.reset_default_graph()

# Where we save our checkpoints and graphs
experiment_dir = os.path.abspath("./experiments_1120/{}".format('tetris'))

# Create a glboal step variable
global_step = tf.Variable(0, name='global_step', trainable=False)

# Create estimators
q_estimator = Estimator(scope="q", summaries_dir=experiment_dir)
target_estimator = Estimator(scope="target_q")

# State processor

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for t, stats in deep_q_learning(sess,
                                    env,
                                    q_estimator=q_estimator,
                                    target_estimator=target_estimator,
                                    experiment_dir=experiment_dir,
                                    num_episodes=100000,
                                    replay_memory_size=500000,
                                    replay_memory_init_size=50000,
                                    update_target_estimator_every=10000,
                                    epsilon_start=1.0,
                                    epsilon_end=0.1,
                                    epsilon_decay_steps=500000,
                                    discount_factor=0.99,
                                    batch_size=32):

        print("\nEpisode Reward: {}".format(stats.episode_rewards[-1]))
 #       print("\nEpisode Line Sent: {}".format(stats.episode_line_sent[-1]))

