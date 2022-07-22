import math

epsilon = 1e-5
INFRACTION_FLAG = -1000

class EvaluationError(Exception):
	def __init__(self, message, errors = None):
		super().__init__(message)
		self.errors = errors

class Evaluator:
	def __init__(self, env):
		env.max_steps = math.inf
		self._env = env
		self._log = {}
		self._score = 0
		self.total_score = 0

		self._last_tiles = [self._env.current_tile()] * 2

	def reward(self, simulator_return) -> float:
		simulator_score = simulator_return[1]

		if simulator_score == -1000:
			raise EvaluationError("")

		self._score = simulator_score

		if self._score < epsilon:
			self._score = 0

		bonus = self.bonus()
		self.total_score += self._score + bonus
		return (self._score + bonus)

	def bonus(self) -> float:
		amount = 0

		current_tile = self._env.current_tile()
		if not (current_tile in self._last_tiles):
			self._last_tiles[1] = self._last_tiles[0]
			self._last_tiles[0] = current_tile
			amount += 10

		amount += self._env.speed/2

		return amount
