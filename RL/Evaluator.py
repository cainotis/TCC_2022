from duckievillage import Evaluator as Base
import math

epsilon = 1e-5

class EvaluationError(Exception):
	def __init__(self, message, errors = None):
		super().__init__(message)
		self.errors = errors

class Evaluator(Base):
	def __init__(self, env):
		env.max_steps = math.inf
		self._env = env
		self._log = {}
		self._score = 0

	def infraction(self, t: str, penalty: float, warning: str):
		print(warning)
		if t not in self._log: self._log[t] = []
		self._log[t].append((penalty, warning))
		raise EvaluationError(warning)

	def track(self):
		r = self._env.penalization(self._env.cur_pos, self._env.cur_angle)
		if r is None: return
		if r == "out":
			self.infraction(r, -1, "Mailduck has gone off-road!")
		if r == "crash":
			self.infraction(r, -1, "Mailduck has crashed into something!")

	def reward(self):
		try:
			self.track()
			self._score = (self._env.speed + self._score) / 2
			if self._score < epsilon:
				self._score = 0

		except EvaluationError as e:
			self._score = -1

		finally:
			return self._score
		
