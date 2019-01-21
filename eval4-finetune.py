#!/usr/bin/env python
# -*- coding: utf-8 -*-
# from train.py
import tensorflow as tf
import threading
import numpy as np

import signal
import random
import os

from network import ActorCriticFFNetwork
from training_thread import A3CTrainingThread

from utils.ops import log_uniform
from utils.rmsprop_applier import RMSPropApplier

from constants import ACTION_SIZE
from constants import PARALLEL_SIZE
from constants import INITIAL_ALPHA_LOW
from constants import INITIAL_ALPHA_HIGH
from constants import INITIAL_ALPHA_LOG_RATE
# from constants import MAX_TIME_STEP
MAX_TIME_STEP = 1*10**6
from constants import CHECKPOINT_DIR
from constants import LOG_FILE
from constants import RMSP_EPSILON
from constants import RMSP_ALPHA
from constants import GRAD_NORM_CLIP
from constants import USE_GPU
from constants import TASK_TYPE
# from constants import TASK_LIST
skipStep=50
# NUM_EVAL_EPISODES = 100
MAX_STEPS = 1000
import random

import pandas as pd

TASK_LIST = {
  'bathroom_02'    : map(str, range(0, 180,skipStep)),
  'bedroom_04'     : map(str, range(0, 408,skipStep)),
  'kitchen_02'     : map(str, range(0, 676,skipStep)),#676
  'living_room_08' : map(str, range(0, 468,skipStep))
}

import pdb

if __name__ == '__main__':

  numSplits=10
  for i in range(0,numSplits):
    # TASK_LIST = {
    #   'bathroom_02'    : map(str, range(i, 180,180//numSplits)),
    #   'bedroom_04'     : map(str, range(i, 408,408//numSplits)),
    #   'kitchen_02'     : map(str, range(i, 676,676//numSplits)),
    #   'living_room_08' : map(str, range(i, 468,468//numSplits))
    # }
    TASK_LIST = {
      'bathroom_02'    : set(list(map(str, random.sample(range(0, 180,1),5)))+['26', '37', '43', '53', '69']),
      'bedroom_04'     : set(list(map(str, random.sample(range(0, 408,1),5)))+['134', '264', '320', '384', '387']),
      'kitchen_02'     : set(list(map(str, random.sample(range(0, 676,1),5)))+['90', '136', '157', '207', '329']),
      'living_room_08' : set(list(map(str, random.sample(range(0, 468,1),5)))+['92', '135', '193', '228', '254'])
    }


    device = "/gpu:0" if USE_GPU else "/cpu:0"
    network_scope = TASK_TYPE
    list_of_tasks = TASK_LIST
    scene_scopes = list_of_tasks.keys()
    global_t = 0
    stop_requested = False

    if not os.path.exists(CHECKPOINT_DIR):
      os.mkdir(CHECKPOINT_DIR)

    initial_learning_rate = log_uniform(INITIAL_ALPHA_LOW,
                                        INITIAL_ALPHA_HIGH,
                                        INITIAL_ALPHA_LOG_RATE)

    global_network = ActorCriticFFNetwork(action_size = ACTION_SIZE,
                                          device = device,
                                          network_scope = network_scope,
                                          scene_scopes = scene_scopes)

    branches = []
    for scene in scene_scopes:
      for task in list_of_tasks[scene]:
        branches.append((scene, task))

    NUM_TASKS = len(branches)
    assert PARALLEL_SIZE >= NUM_TASKS, \
      "Not enough threads for multitasking: at least {} threads needed.".format(NUM_TASKS)

    learning_rate_input = tf.placeholder("float")
    grad_applier = RMSPropApplier(learning_rate = learning_rate_input,
                                  decay = RMSP_ALPHA,
                                  momentum = 0.0,
                                  epsilon = RMSP_EPSILON,
                                  clip_norm = GRAD_NORM_CLIP,
                                  device = device)

    # instantiate each training thread
    # each thread is training for one target in one scene
    training_threads = []
    for i in range(PARALLEL_SIZE):
      scene, task = branches[i%NUM_TASKS]
      training_thread = A3CTrainingThread(i, global_network, initial_learning_rate,
                                          learning_rate_input,
                                          grad_applier, MAX_TIME_STEP,
                                          device = device,
                                          network_scope = "thread-%d"%(i+1),
                                          scene_scope = scene,
                                          task_scope = task)
      training_threads.append(training_thread)

    # prepare session
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False,
                                            allow_soft_placement=True))

    init = tf.global_variables_initializer()
    sess.run(init)

    # create tensorboard summaries
    summary_op = dict()
    summary_placeholders = dict()

    for i in range(PARALLEL_SIZE):
      scene, task = branches[i%NUM_TASKS]
      key = scene + "-" + task

      # summary for tensorboard
      episode_reward_input = tf.placeholder("float")
      episode_length_input = tf.placeholder("float")
      episode_max_q_input  = tf.placeholder("float")

      scalar_summaries = [
        tf.summary.scalar(key+"/Episode_Reward", episode_reward_input),
        tf.summary.scalar(key+"/Episode_Length", episode_length_input),
        tf.summary.scalar(key+"/Episode_Max_Q", episode_max_q_input)
      ]

      summary_op[key] = tf.summary.merge(scalar_summaries)
      summary_placeholders[key] = {
        "episode_reward_input": episode_reward_input,
        "episode_length_input": episode_length_input,
        "episode_max_q_input": episode_max_q_input,
        "learning_rate_input": learning_rate_input
      }

    summary_writer = tf.summary.FileWriter(LOG_FILE, sess.graph)

    # init or load checkpoint with saver
    # if you don't need to be able to resume training, use the next line instead.
    # it will result in a much smaller checkpoint file.
    saver = tf.train.Saver(max_to_keep=10, var_list=global_network.get_vars())

    var_list=tf.trainable_variables()#global_network.get_vars()




    #saver = tf.train.Saver(max_to_keep=10)




    checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
    if checkpoint and checkpoint.model_checkpoint_path:
      saver.restore(sess, checkpoint.model_checkpoint_path)
      print("checkpoint loaded: {}".format(checkpoint.model_checkpoint_path))
      tokens = checkpoint.model_checkpoint_path.split("-")
      # set global step
      global_t = int(tokens[1])
      print(">>> global step set: {}".format(global_t))
    else:
      print("Could not find old checkpoint")


    def train_function(parallel_index):
      global global_t
      training_thread = training_threads[parallel_index]
      last_global_t = 0

      scene, task = branches[parallel_index % NUM_TASKS]
      key = scene + "-" + task

      while global_t < MAX_TIME_STEP and not stop_requested:
        diff_global_t = training_thread.process(sess, global_t, summary_writer,
                                                summary_op[key], summary_placeholders[key])
        global_t += diff_global_t
        # periodically save checkpoints to disk
        if parallel_index == 0 and global_t - last_global_t > 1000000:
          # print('Save checkpoint at timestamp %d' % global_t)
          # saver.save(sess, CHECKPOINT_DIR + '/' + 'checkpoint', global_step = global_t)
          last_global_t = global_t

    def signal_handler(signal, frame):
      global stop_requested
      print('You pressed Ctrl+C!')
      stop_requested = True

    train_threads = []
    for i in range(PARALLEL_SIZE):
      train_threads.append(threading.Thread(target=train_function, args=(i,)))

    signal.signal(signal.SIGINT, signal_handler)

    # start each training thread
    for t in train_threads:
      t.start()

    print('Press Ctrl+C to stop.')
    signal.pause()

    # wait for all threads to finish
    for t in train_threads:
      t.join()

    # print('Now saving data. Please wait.')
    # saver.save(sess, CHECKPOINT_DIR + '/' + 'checkpoint', global_step = global_t)
    summary_writer.close()












#from eval2.py
#!/usr/bin/env python
# # -*- coding: utf-8 -*-
# import tensorflow as tf
# import numpy as np
# import random
# import sys

# from network import ActorCriticFFNetwork
# from training_thread import A3CTrainingThread
# from scene_loader import THORDiscreteEnvironment as Environment

# from utils.ops import sample_action

# from constants import ACTION_SIZE
# from constants import CHECKPOINT_DIR
# from constants import NUM_EVAL_EPISODES
# from constants import VERBOSE

# from constants import TASK_TYPE
# # from constants import TASK_LIST
# #180,408,676,468
# skipStep=1
# NUM_EVAL_EPISODES = 100
# MAX_STEPS = 1000

# import pandas as pd

# TASK_LIST = {
#   'bathroom_02'    : map(str, range(0, 180,skipStep)),
#   'bedroom_04'     : map(str, range(0, 408,skipStep)),
#   'kitchen_02'     : map(str, range(0, 676,skipStep)),#676
#   'living_room_08' : map(str, range(0, 468,skipStep))
# }


# if __name__ == '__main__':

#   device = "/cpu:0" # use CPU for display tool
#   network_scope = TASK_TYPE
#   list_of_tasks = TASK_LIST
#   scene_scopes = list_of_tasks.keys()

#   global_network = ActorCriticFFNetwork(action_size=ACTION_SIZE,
#                                         device=device,
#                                         network_scope=network_scope,
#                                         scene_scopes=scene_scopes)

#   sess = tf.Session()
#   init = tf.global_variables_initializer()
#   sess.run(init)

#   saver = tf.train.Saver()
#   checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_DIR)

#   if checkpoint and checkpoint.model_checkpoint_path:
#     saver.restore(sess, checkpoint.model_checkpoint_path)
#     print("checkpoint loaded: {}".format(checkpoint.model_checkpoint_path))
#   else:
#     print("Could not find old checkpoint")

#   resultList = []
#   allResultList = []
#   scene_stats = dict()
#   for scene_scope in scene_scopes:

#     scene_stats[scene_scope] = []
#     for task_scope in list_of_tasks[scene_scope]:

#       print(task_scope)
#       print("______________________________________________")

#       env = Environment({
#         'scene_name': scene_scope,
#         'terminal_state_id': int(task_scope)
#       })
#       ep_rewards = []
#       ep_lengths = []
#       ep_collisions = []

#       scopes = [network_scope, scene_scope, task_scope]

#       for i_episode in range(NUM_EVAL_EPISODES):

#         env.reset()
#         terminal = False
#         ep_reward = 0
#         ep_collision = 0
#         ep_t = 0

#         while not terminal:

#           pi_values = global_network.run_policy(sess, env.s_t, env.target, scopes)
#           action = sample_action(pi_values)
#           env.step(action)
#           env.update()

#           terminal = env.terminal
#           if ep_t == MAX_STEPS: break
#           if env.collided: ep_collision += 1
#           ep_reward += env.reward
#           ep_t += 1

#         ep_lengths.append(ep_t)
#         ep_rewards.append(ep_reward)
#         ep_collisions.append(ep_collision)
#         rs = [scene_scope,task_scope,ep_reward,ep_t,ep_collision]
#         allResultList.append(rs)
#         if VERBOSE: print("episode #{} ends after {} steps".format(i_episode, ep_t))

#       print('evaluation: %s %s' % (scene_scope, task_scope))
#       print('mean episode reward: %.2f' % np.mean(ep_rewards))
#       print('mean episode length: %.2f' % np.mean(ep_lengths))
#       print('mean episode collision: %.2f' % np.mean(ep_collisions))
#       results = [scene_scope,task_scope,np.mean(ep_rewards),np.mean(ep_lengths),np.mean(ep_collisions)]
#       resultList.append(results)

#       scene_stats[scene_scope].extend(ep_lengths)
#       pd.DataFrame(resultList).to_csv('intermediate-eval4.csv')
#       #break
#     #break

# print('\nResults (average trajectory length):')
# df = pd.DataFrame(resultList)
# df.columns = ["scene_scope","task_scope","ep_rewards","ep_lengths","ep_collisions"]
# df.to_csv('eval4.csv')
# pd.DataFrame(allResultList).to_csv('eval4-allResults.csv')
# for scene_scope in scene_stats:
#   print('%s: %.2f steps'%(scene_scope, np.mean(scene_stats[scene_scope])))

