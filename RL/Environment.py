from typing import Optional

from .DuckievillageEnv import DuckievillageEnv

from .Evaluator import Evaluator

import abc
import tensorflow as tf
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts


import cv2


class Environment(DuckievillageEnv):
	def __init__(self,
				 interative: Optional[bool] = False,
				 **kwargs):
		super().__init__(**kwargs)
		self.eval = Evaluator(self)
		self._current_time_step = None
		self._interative = True
		self._action_spec = array_spec.BoundedArraySpec(
			shape=(2,),
			dtype=np.float32,
			minimum=0,
			maximum=1,
			name='action'
		)
		self._observation_spec = array_spec.BoundedArraySpec(
			shape=(60, 80, 3),
			dtype=np.int32,
			minimum=0,
			maximum=255,
			name='observation'
		)

		self._episode_ended = False

	def _reset(self):
		"""Return initial_time_step."""
		super().reset()
		return ts.restart(np.array([None], dtype=np.ndarray))
		# return ts.restart(np.array([self._state()], dtype=np.ndarray))

	def _step(self, action):

		if self._episode_ended:
			# The last action ended the episode. Ignore the current action and start
			# a new episode.
			return self.reset()

		pwm_left, pwm_right = action
		ret = super().step(pwm_left=pwm_left, pwm_right=pwm_right)
		
		# Refresh at every update.
		if self._interative:
			self.render()

		reward = self.eval.reward()

		if reward == -1:
			return ts.termination(
				np.array([self._state()], dtype=np.ndarray), self.eval.total_score)

		return ts.transition(
			np.array([self._state()], dtype=np.ndarray),
			reward=reward,
			discount=1.0
		)

	def _state(self):
		I = cv2.resize(self.front(), (80, 60))/255
		return I.reshape((-1, 60, 80, 3))

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
