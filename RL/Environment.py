from typing import Optional

from .BaseEnvironment import BaseEnvironment

import abc
import tensorflow as tf
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts

import cv2

OBSERVATION_SHAPE = (3, )

class Environment(BaseEnvironment, py_environment.PyEnvironment):
	def __init__(self,
				 interative: Optional[bool] = False,
				 **kwargs):
		super().__init__(**kwargs)
		self._current_time_step = None
		self._interative = interative
		self._action_spec = array_spec.BoundedArraySpec(
			shape=(2,),
			dtype=np.int64,
			minimum=0,
			maximum=8,
			name='action'
		)
		self._observation_spec = array_spec.BoundedArraySpec(
			shape=OBSERVATION_SHAPE,
			dtype=np.float16,
			minimum=[0, 0, -np.pi],
			maximum=[4, 4, np.pi],
			name='observation'
		)

		self._episode_ended = False

		self._action2pwm = [-1, -.75, -.5, -.25, 0, .25, .5, .75, 1]

	def _reset(self):
		"""Return initial_time_step."""
		super().reset()
		# return ts.restart(np.array([None], dtype=np.ndarray))
		return ts.restart(
			self._state()
		)

	def _step(self, action):

		if self._episode_ended:
			# The last action ended the episode. Ignore the current action and start
			# a new episode.
			return self.reset()

		pwm_left = self._action2pwm[action[0]]

		pwm_right = self._action2pwm[action[1]]

		ret = super().step(pwm_left=pwm_left, pwm_right=pwm_right)
		
		# Refresh at every update.
		if self._interative:
			self.render()

		reward = self.eval.reward(ret)

		if reward == -1000:
			return ts.termination(
				self._state(),
				self.eval.total_score
			)

		return ts.transition(
			self._state(),
			reward=reward,
			discount=0
		)

	def _state(self):
		x, y, z = self.cur_pos
		angle = self.cur_angle
		return np.float16([x, z, angle])

	def current_time_step(self):
		return self._current_time_step

	def action_spec(self):
		return self._action_spec

	def observation_spec(self):
		return self._observation_spec

	def reset(self):
		"""Return initial_time_step."""
		self._current_time_step = self._reset()
		return self._current_time_step

	def step(self, action):
		"""Apply action and return new time_step."""
		if self._current_time_step is None:
			return self.reset()
		self._current_time_step = self._step(action)
		return self._current_time_step