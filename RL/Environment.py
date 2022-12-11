from typing import Optional

from .BaseEnvironment import BaseEnvironment
from .Evaluator import EvaluationError

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

OBSERVATION_SHAPE = (2, )

class Environment(BaseEnvironment, py_environment.PyEnvironment):
	def __init__(self,
				 interative: Optional[bool] = False,
				 **kwargs):
		super().__init__(**kwargs)

		self._current_time_step = None
		self._interative = interative

		self._action2move = [
			(0, -20),

			(0.75, 0),

			(0, 20),
		]
		
		self._action_spec = array_spec.BoundedArraySpec(
			shape=(),
			dtype=np.int32,
			minimum=0,
			maximum=len(self._action2move)-1,
			name='action'
		)

		self._observation_spec = array_spec.BoundedArraySpec(
			shape=OBSERVATION_SHAPE,
			dtype=np.float64,
			minimum=[-1, -90],
			maximum=[1, 90],
			name='observation'
		)

		self._episode_ended = False
		self._last_step

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

		vel, angle = self._action2move[action]

		ret = super().step(vel=vel, angle=angle)
		
		# Refresh at every update.
		if self._interative:
			self.render()

		try :
			reward = self.eval.reward(ret)
		except EvaluationError:
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
		try :
			aux = self.get_lane_pos2(self.cur_pos, self.cur_angle)
			ret = np.float64([aux.dist, aux.angle_deg])
			self._last_step = ret
		except:
			ret = self._last_step
		return ret

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
