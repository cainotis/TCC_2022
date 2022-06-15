import sys
import pyglet
from pyglet.window import key

from RL import Environment, EvaluationError

import tensorflow as tf

from tf_agents.environments import tf_py_environment

from tf_agents.environments import utils

def main():
	env = Environment(
		seed = 101,
		# map_name = 'loop_empty',
		draw_curve = False,
		draw_bbox = False,
		domain_rand = False,
		distortion = False,
		top_down = False,

		# enable_eval = True,

		map_name = 'loop_empty',
		is_external_map = False,
	)

	# env.reset()

	utils.validate_py_environment(env, episodes=5)


if __name__ == '__main__':
	main()