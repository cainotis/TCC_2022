#https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial

from __future__ import absolute_import, division, print_function

import logging
from logging.handlers import RotatingFileHandler

file_handler = RotatingFileHandler("logs/tcc.log", maxBytes=1000000, backupCount=5)
file_handler.setLevel(logging.DEBUG)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.WARN)

logging.basicConfig(
	handlers=[
		file_handler, stream_handler
	],
	format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
	datefmt='%H:%M:%S',
	level=logging.DEBUG
)


import time
import base64
import imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import reverb

import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import py_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import sequential
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.utils import common


from tf_agents.environments import utils
from tf_agents.environments import TimeLimit
from tf_agents.policies import PolicySaver

from RL import Environment, EvaluationError
from datetime import datetime

time2stop = datetime(2022, 8, 30, 23, 17, 0, 0)

initial_collect_steps = 100  # @param {type:"integer"}
collect_steps_per_iteration = 1 # @param {type:"integer"}
replay_buffer_max_length = 100000  # @param {type:"integer"}

batch_size = 64  # @param {type:"integer"}
learning_rate = 1e-3  # @param {type:"number"}
log_interval = 200  # @param {type:"integer"}

num_eval_episodes = 10  # @param {type:"integer"}
eval_interval = 1000  # @param {type:"integer"}

env = Environment(
	seed = 101,
	draw_curve = False,
	draw_bbox = False,
	domain_rand = False,
	distortion = False,
	top_down = False,

	map_name = 'loop_empty',
)
env = TimeLimit(env=env, duration = 1000)
env.reset()

train_py_env = Environment(
	seed = 101,
	draw_curve = False,
	draw_bbox = False,
	domain_rand = False,
	distortion = False,
	top_down = False,

	map_name = 'loop_empty',
)

eval_py_env = Environment(
	seed = 101,
	draw_curve = False,
	draw_bbox = False,
	domain_rand = False,
	distortion = False,
	top_down = False,

	map_name = 'loop_empty',
)

train_py_env = TimeLimit(env=train_py_env, duration = 2000)
eval_py_env = TimeLimit(env=eval_py_env, duration = 2000)

train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

fc_layer_params = (100, 75, 50)
action_tensor_spec = tensor_spec.from_spec(env.action_spec())
num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1

# Define a helper function to create Dense layers configured with the right
# activation and kernel initializer.
def dense_layer(num_units):
	return tf.keras.layers.Dense(
			num_units,
			activation=tf.keras.activations.relu,
			kernel_initializer=tf.keras.initializers.VarianceScaling(
					scale=2.0, mode='fan_in', distribution='truncated_normal'))

# QNetwork consists of a sequence of Dense layers followed by a dense layer
# with `num_actions` units to generate one q_value per available action as
# its output.
dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]
q_values_layer = tf.keras.layers.Dense(
		num_actions,
		activation=None,
		kernel_initializer=tf.keras.initializers.RandomUniform(
				minval=-0.03, maxval=0.03),
		bias_initializer=tf.keras.initializers.Constant(-0.2))
q_net = sequential.Sequential(dense_layers + [q_values_layer])


optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

train_step_counter = tf.Variable(0)

agent = dqn_agent.DqnAgent(
		train_env.time_step_spec(),
		train_env.action_spec(),
		q_network=q_net,
		optimizer=optimizer,
		td_errors_loss_fn=common.element_wise_squared_loss,
		train_step_counter=train_step_counter)

agent.initialize()


eval_policy = agent.policy
collect_policy = agent.collect_policy


initial_policy = random_tf_policy.RandomTFPolicy(
	train_env.time_step_spec(),
	train_env.action_spec()
)

example_environment = Environment(
	seed = 101,
	draw_curve = False,
	draw_bbox = False,
	domain_rand = False,
	distortion = False,
	top_down = False,

	map_name = 'loop_empty',
)
example_environment = TimeLimit(env=example_environment, duration = 1000)
example_environment = tf_py_environment.TFPyEnvironment(example_environment)

time_step = example_environment.reset()

initial_policy.action(time_step)

def compute_avg_return(environment, policy, num_episodes=10):

	total_return = 0.0
	for _ in range(num_episodes):

		time_step = environment.reset()
		episode_return = 0.0

		while not time_step.is_last():
			action_step = policy.action(time_step)
			time_step = environment.step(action_step.action)
			episode_return += time_step.reward
		total_return += episode_return

	avg_return = total_return / num_episodes
	return avg_return.numpy()[0]


# See also the metrics module for standard implementations of different metrics.
# https://github.com/tensorflow/agents/tree/master/tf_agents/metrics

compute_avg_return(eval_env, initial_policy, num_eval_episodes)


table_name = 'uniform_table'
replay_buffer_signature = tensor_spec.from_spec(
			agent.collect_data_spec)
replay_buffer_signature = tensor_spec.add_outer_dim(
		replay_buffer_signature)

table = reverb.Table(
		table_name,
		max_size=replay_buffer_max_length,
		sampler=reverb.selectors.Uniform(),
		remover=reverb.selectors.Fifo(),
		rate_limiter=reverb.rate_limiters.MinSize(1),
		signature=replay_buffer_signature)

reverb_server = reverb.Server([table])

replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
		agent.collect_data_spec,
		table_name=table_name,
		sequence_length=2,
		local_server=reverb_server)

rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
	replay_buffer.py_client,
	table_name,
	sequence_length=2)

py_driver.PyDriver(
		env,
		py_tf_eager_policy.PyTFEagerPolicy(
			initial_policy, use_tf_function=True),
		[rb_observer],
		max_steps=initial_collect_steps).run(train_py_env.reset())

# Dataset generates trajectories with shape [Bx2x...]
dataset = replay_buffer.as_dataset(
		num_parallel_calls=3,
		sample_batch_size=batch_size,
		num_steps=2).prefetch(3)

dataset

iterator = iter(dataset)
print(iterator)


# (Optional) Optimize by wrapping some of the code in a graph using TF function.
agent.train = common.function(agent.train)

# Reset the train step.
agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
returns = [avg_return]

# Reset the environment.
time_step = train_py_env.reset()

# Create a driver to collect experience.
collect_driver = py_driver.PyDriver(
		env,
		py_tf_eager_policy.PyTFEagerPolicy(
			agent.collect_policy, use_tf_function=True),
		[rb_observer],
		max_steps=collect_steps_per_iteration)

## PolicySaver
my_policy = agent.collect_policy
saver = PolicySaver(my_policy, batch_size=None)

## Checkpoiter
global_step = tf.Variable(agent.train_step_counter.numpy(), name="step")
train_checkpointer = common.Checkpointer(
    ckpt_dir='checkpoint',
    max_to_keep=10,
    agent=agent,
    policy=agent.policy,
    replay_buffer=replay_buffer,
    global_step=global_step
)
train_checkpointer.initialize_or_restore()
global_step = tf.compat.v1.train.get_global_step()

num_iterations = 0

while datetime.now() < time2stop:
	num_iterations += 1
	# Collect a few steps and save to the replay buffer.
	time_step, _ = collect_driver.run(time_step)

	# Sample a batch of data from the buffer and update the agent's network.
	experience, unused_info = next(iterator)
	train_loss = agent.train(experience).loss

	step = agent.train_step_counter.numpy()	

	if step % log_interval == 0:
		print('step = {0}: loss = {1}'.format(step, train_loss))

	if step % eval_interval == 0:
		avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
		print('step = {0}: Average Return = {1}'.format(step, avg_return))
		returns.append(avg_return)
		saver.save(f'policies/policy_{time.strftime("%Y%m%d-%H%M%S")}_{step}')
		train_checkpointer.save(step)
		# print(f"now: {datetime.now().strftime('%Y/%m/%d, %H:%M:%S')}")
		print(f"stop time : {time2stop.strftime('%Y/%m/%d, %H:%M:%S')}")

step = agent.train_step_counter.numpy()
saver.save(f'policies/policy_{time.strftime("%Y%m%d-%H%M%S")}_{step}')
train_checkpointer.save(step)

iterations = range(0, num_iterations + 1, eval_interval)
plt.plot(iterations, returns)
plt.ylabel('Average Return')
plt.xlabel('Iterations')
plt.ylim(top=250)

def create_policy_eval_video(policy, filename, num_episodes=5, fps=30):
	filename = filename + ".mp4"
	with imageio.get_writer(filename, fps=fps) as video:
		for _ in range(num_episodes):
			time_step = eval_env.reset()
			video.append_data(eval_py_env.render())
			while not time_step.is_last():
				action_step = policy.action(time_step)
				time_step = eval_env.step(action_step.action)
				video.append_data(eval_py_env.render())

create_policy_eval_video(agent.policy, "trained-agent")