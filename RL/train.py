import sys
import pyglet
from pyglet.window import key

import matplotlib.pyplot as plt

import tensorflow as tf

from tf_agents.agents.reinforce import reinforce_agent

from tf_agents.drivers import py_driver

from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils

from tf_agents.networks import actor_distribution_network

from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import PolicySaver

from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils

from tf_agents.specs import tensor_spec

from tf_agents.trajectories import trajectory

from tf_agents.utils import common


from RL import Environment, EvaluationError

from RL.utils import next_path


num_iterations = 2 #50 # @param {type:"integer"}
collect_episodes_per_iteration = 2 # @param {type:"integer"}
replay_buffer_capacity = 2000 # @param {type:"integer"}

fc_layer_params = (100,)

learning_rate = 1e-3 # @param {type:"number"}
log_interval = 25 # @param {type:"integer"}
num_eval_episodes = 10 # @param {type:"integer"}
eval_interval = 50 # @param {type:"integer"}


def main():
	train_py_env = Environment(
		seed = 101,
		draw_curve = False,
		draw_bbox = False,
		domain_rand = False,
		distortion = False,
		top_down = False,

		map_name = 'loop_empty',
		is_external_map = False,
		interative = True,
	)

	train_py_env.reset()

	eval_py_env = Environment(
		
		seed = 101,
		draw_curve = False,
		draw_bbox = False,
		domain_rand = False,
		distortion = False,
		top_down = False,

		map_name = 'maps/circuit.yaml',
		is_external_map = True,
		
		enable_eval = True,
		interative = True,
	)


	# utils.validate_py_environment(train_py_env, episodes=5)
	# utils.validate_py_environment(eval_py_env, episodes=5)

	train_env = tf_py_environment.TFPyEnvironment(train_py_env)
	eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

	# Agent

	actor_net = actor_distribution_network.ActorDistributionNetwork(
		train_env.observation_spec(),
		train_env.action_spec(),
		fc_layer_params=fc_layer_params
	)

	optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

	train_step_counter = tf.Variable(0)

	tf_agent = reinforce_agent.ReinforceAgent(
		train_env.time_step_spec(),
		train_env.action_spec(),
		actor_network=actor_net,
		optimizer=optimizer,
		normalize_returns=True,
		train_step_counter=train_step_counter
	)
	tf_agent.initialize()

	# Policies

	eval_policy = tf_agent.policy
	collect_policy = tf_agent.collect_policy

	saver = PolicySaver(eval_policy, batch_size=None)

	# (Optional) Optimize by wrapping some of the code in a graph using TF function.
	tf_agent.train = common.function(tf_agent.train)

	# Reset the train step
	tf_agent.train_step_counter.assign(0)

	# Evaluate the agent's policy once before training.
	avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
	returns = [avg_return]

	for _ in range(num_iterations):

		# Collect a few episodes using collect_policy and save to the replay buffer.
		# collect_episode(
		# train_py_env, tf_agent.collect_policy, collect_episodes_per_iteration)

		# Use data from the buffer and update the agent's network.
		iterator = iter(replay_buffer.as_dataset(sample_batch_size=1))
		trajectories, _ = next(iterator)
		train_loss = tf_agent.train(experience=trajectories)  

		replay_buffer.clear()

		step = tf_agent.train_step_counter.numpy()

		if step % log_interval == 0:
			print('step = {0}: loss = {1}'.format(step, train_loss.loss))

		if step % eval_interval == 0:
			avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
			print('step = {0}: Average Return = {1}'.format(step, avg_return))
			returns.append(avg_return)

		saver.save(next_path('policy_%s'))
	
	## Visualization
	steps = range(0, num_iterations + 1, eval_interval)
	plt.plot(steps, returns)
	plt.ylabel('Average Return')
	plt.xlabel('Step')
	plt.ylim(top=250)

	num_episodes = 3
	video_filename = 'imageio.mp4'
	with imageio.get_writer(video_filename, fps=60) as video:
		for _ in range(num_episodes):
			time_step = eval_env.reset()
			video.append_data(eval_py_env.render())
			while not time_step.is_last():
				action_step = tf_agent.policy.action(time_step)
				time_step = eval_env.step(action_step.action)
				video.append_data(eval_py_env.render())

	
## Metric
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

## Repley 
##		reverb not working


## Data Collection

# def collect_episode(environment, policy, num_episodes):

# 	driver = py_driver.PyDriver(
# 		environment,
# 		py_tf_eager_policy.PyTFEagerPolicy(
# 			policy, 
# 			use_tf_function=True
# 		),
# 		max_episodes=num_episodes
# 	)
# 	initial_time_step = environment.reset()
# 	driver.run(initial_time_step)

if __name__ == '__main__':
	main()