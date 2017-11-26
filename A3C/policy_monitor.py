import sys
import os
import itertools
import collections
import numpy as np
import tensorflow as tf
import time

from inspect import getsourcefile
current_path = os.path.dirname(os.path.abspath(getsourcefile(lambda:0)))
import_path = os.path.abspath(os.path.join(current_path, "../.."))

if import_path not in sys.path:
    sys.path.append(import_path)

#from gym.wrappers import Monitor
#import gym

#from lib.atari.state_processor import StateProcessor
#from lib.atari import helpers as atari_helpers
from estimators import ValueEstimator, PolicyEstimator
from worker import make_copy_params_op
from tetris import Tetris,Screen

class PolicyMonitor(object):
    """
    Helps evaluating a policy by running an episode in an environment,
    saving a video, and plotting summaries to Tensorboard.

    Args:
    env: environment to run in
    policy_net: A policy estimator
    summary_writer: a tf.train.SummaryWriter used to write Tensorboard summaries
    """
    def __init__(self, env, policy_net, summary_writer, saver=None):

        #self.video_dir = os.path.join(summary_writer.get_logdir(), "../videos")
    #self.video_dir = os.path.abspath(self.video_dir)

    #self.env = Monitor(env, directory=self.video_dir, video_callable=lambda x: True, resume=True)
        self.env = env
        self.global_policy_net = policy_net
        self.summary_writer = summary_writer
        self.saver = saver
        #self.sp = StateProcessor()

        self.checkpoint_path = os.path.abspath(os.path.join(summary_writer.get_logdir(), "../checkpoints/model"))


        # Local policy net
        with tf.variable_scope("policy_eval"):
            self.policy_net = PolicyEstimator(policy_net.num_outputs)

        # Op to copy params from global policy/value net parameters
        self.copy_params_op = make_copy_params_op(
            tf.contrib.slim.get_variables(scope="global", collection=tf.GraphKeys.TRAINABLE_VARIABLES),
            tf.contrib.slim.get_variables(scope="policy_eval", collection=tf.GraphKeys.TRAINABLE_VARIABLES))

        def _policy_net_predict(self, state, sess):
            feed_dict = { self.policy_net.grid_states: [state[0]],
                         self.policy_net.piece_states: [state[1]]}
        preds = sess.run(self.policy_net.predictions, feed_dict)
    return preds["probs"][0]

    def eval_once(self, sess):
        with sess.as_default(), sess.graph.as_default():
            # Copy params to local model
            global_step, _ = sess.run([tf.contrib.framework.get_global_step(), self.copy_params_op])

            # Run an episode
            done = False
            #state = atari_helpers.atari_make_initial_state(self.sp.process(self.env.reset()))
            state =self.env.reset()
            total_reward = 0.0
            episode_length = 0
            total_sent = 0
            total_cleared = 0
            while not done:
                action_probs = self._policy_net_predict(state, sess)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, line_sent, line_cleared = self.env.step(action)
            #next_state = atari_helpers.atari_make_next_state(state, self.sp.process(next_state))
            total_reward += reward
            episode_length += 1
            state = next_state
            total_sent += line_sent
            total_cleared += line_cleared

            # Add summaries
            episode_summary = tf.Summary()
            episode_summary.value.add(simple_value=total_reward, tag="eval/total_reward")
            episode_summary.value.add(simple_value=episode_length, tag="eval/episode_length")
            episode_summary.value.add(simple_value=total_sent, tag="eval/total_sent")
            episode_summary.value.add(simple_value=total_cleared, tag="eval/total_cleared")
            self.summary_writer.add_summary(episode_summary, global_step)
            self.summary_writer.flush()

            if self.saver is not None:
                self.saver.save(sess, self.checkpoint_path)

        tf.logging.info("Eval results at step {}: total_reward {}, episode_length {}, total_sent {}".format(global_step, total_reward, episode_length,total_sent))

        return total_reward, episode_length, total_sent,total_cleared
    
    def continuous_eval(self, eval_every, sess, coord):
      """
      Continuously evaluates the policy every [eval_every] seconds.
      """
      game_length = []
      total_cleared = []
      total_sent = []
      index = 0
      while not coord.should_stop():
            index +=1
            _,length,sent,cleared = self.eval_once(sess)
            game_length.append(length)
            total_cleared.append(cleared)
            total_sent.append(sent)
            print("Test " + str(index) + ": Total line cleared : " + str(cleared) + " Total lines sent : " + str(sent) + " Total length:" + str(length))

            if index >= 100:
                break
            # Sleep until next evaluation cycle
            mean_lines = np.mean(total_sent)
            mean_length = np.mean(game_length)
            mean_cleared = np.mean(total_cleared)
            print("Mean of line cleared : " + str(mean_cleared))
            print("Mean of line sent : " + str(mean_lines))
            print("Mean of game length : " + str(mean_length))
            print("Mean")
        #time.sleep(eval_every)
    return

