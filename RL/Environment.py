from .DuckievillageEnv import DuckievillageEnv

from .Evaluator import Evaluator


class Environment(DuckievillageEnv):
	def __init__(self,
	             enable_eval: bool = True,
	             **kwargs):
		super().__init__(**kwargs)
		self.eval = Evaluator(self) if enable_eval else None