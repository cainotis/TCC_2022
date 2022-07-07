# import logging
import math
import random
from ctypes import POINTER

import numpy as np
import numpy.linalg
import cv2
import gym.wrappers.monitoring.video_recorder
import gym_duckietown.objmesh
import gym_duckietown.objects
import gym_duckietown.simulator
import gym_duckietown.envs
import gym_duckietown.wrappers
import pyglet
from pyglet import gl, window, image

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

FULL_VIEW_MODE = 0
TOP_DOWN_VIEW_MODE = 1
FRONT_VIEW_MODE = 2
N_VIEW_MODES = 3

from .Evaluator import Evaluator

class BaseEnvironment(gym_duckietown.envs.DuckietownEnv):
	def __init__(self,
			 top_down = False,
			 cam_height = 5,
			 video_path: str = None,
			 **kwargs):
		super().__init__(**kwargs)
		self.horizon_color = self._perturb(self.color_sky)
		self.cam_fov_y = gym_duckietown.simulator.CAMERA_FOV_Y
		self.top_down = top_down
	
		self.eval = Evaluator(self)

		self._view_mode = 0
		self.top_cam_height = cam_height

		self.actions = [0, 0]

		if video_path is not None:
			self.rec = gym.wrappers.monitoring.video_recorder.VideoRecorder(self, path=video_path, enabled = True)
		else:
			self.rec = None

		self.renderables = None

	def add_renderable(self, r):
		if self.renderables is None: self.renderables = [r]
		else: self.renderables.append(r)
		return self.renderables

	def next_view(self):
		self._view_mode = (self._view_mode + 1) % N_VIEW_MODES

	def set_view(self, view):
		self._view_mode = view % N_VIEW_MODES

	def toggle_single_view(self):
		if self._view_mode == TOP_DOWN_VIEW_MODE:
			self._view_mode = FRONT_VIEW_MODE
		else:
			self._view_mode = TOP_DOWN_VIEW_MODE

	# Returns a list with all road tiles in the current map. Each item of the list contains the
	# tile indices (not positional coordinates), and the tile type (curve, straight, intersection).
	def roads(self):
		return self._roads

	def current_tile(self):
		return self.get_grid_coords(self.cur_pos)

	def tile_center(self, i, j=None):
		if j is None:
			i, j = i[0], i[1]
		return (np.array([i, j])+0.5)*self.road_tile_size

	def nearest_drivable(self, x: float, y: float = None) -> (float, float):
		if y is None: x, y = x[0], x[1]
		min_d, min_p = math.inf, None
		for t in self.drivable_tiles:
			a, b = t["coords"]
			cx, cy = (a+0.5)*self.road_tile_size, (b+0.5)*self.road_tile_size
			d = (cx-x)*(cx-x)+(cy-y)*(cy-y)
			if min_d > d: min_d, min_p = d, (cx, cy)
		return min_p

	def get_position(self):
		return np.delete(self.cur_pos, 1)

	def top_down_obs(self, segment = False):
		return self._render_img(
			WINDOW_WIDTH,
			WINDOW_HEIGHT,
			self.multi_fbo_human,
			self.final_fbo_human,
			self.img_array_human,
			top_down = True,
			segment=segment,
			# callback = (lambda: self.mailbox.render()) if self.mailbox is not None else None,
		)

	def front(self, segment = False):
		return self._render_img(
			WINDOW_WIDTH,
			WINDOW_HEIGHT,
			self.multi_fbo_human,
			self.final_fbo_human,
			self.img_array_human,
			top_down = False,
			segment=segment,
			# callback = (lambda: self.mailbox.render()) if self.mailbox is not None else None,
		)

	def render(self, mode: str = "human", close: bool = False, segment: bool = False, text: str = ""):
		"""
		Render the environment for human viewing

		mode: "human", "top_down", "free_cam", "rgb_array"

		"""
		assert mode in ["human", "top_down", "free_cam", "rgb_array"]

		if close:
			if self.window:
				self.window.close()
			return

		top_down = mode == 'top_down'
		# Render the image
		top = self.top_down_obs(segment)
		bot = self.front(segment)

		if self.distortion and not self.undistort and mode != "free_cam":
			bot = self.camera_model.distort(bot)

		win_width = WINDOW_WIDTH
		if self._view_mode == FULL_VIEW_MODE:
			img = np.concatenate((top, bot), axis=1)
			win_width = 2*WINDOW_WIDTH
		elif self._view_mode == TOP_DOWN_VIEW_MODE:
			img = top
		else:
			img = bot


		if self.window is not None:
			self.window.set_size(win_width, WINDOW_HEIGHT)

		if mode == 'rgb_array':
			return img

		if self.window is None:
			config = gl.Config(double_buffer=False)
			self.window = window.Window(
				width=win_width,
				height=WINDOW_HEIGHT,
				resizable=False,
				config=config
			)

		self.window.clear()
		self.window.switch_to()
		self.window.dispatch_events()

		# Bind the default frame buffer
		gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

		# Setup orghogonal projection
		gl.glMatrixMode(gl.GL_PROJECTION)
		gl.glLoadIdentity()
		gl.glMatrixMode(gl.GL_MODELVIEW)
		gl.glLoadIdentity()
		gl.glOrtho(0, WINDOW_WIDTH, 0, WINDOW_HEIGHT, 0, 10)

		# Draw the image to the rendering window
		width = img.shape[1]
		height = img.shape[0]
		img = np.ascontiguousarray(np.flip(img, axis=0))
		img_data = image.ImageData(
			width,
			height,
			'RGB',
			img.ctypes.data_as(POINTER(gl.GLubyte)),
			pitch=width * 3,
		)
		img_data.blit(
			0,
			0,
			0,
			width=WINDOW_WIDTH,
			height=WINDOW_HEIGHT
		)

		# Display position/state information
		if mode != "free_cam":
			x, y, z = self.cur_pos
			self.text_label.text = (
				f"pos: ({x:.2f}, {y:.2f}, {z:.2f}), angle: "
				f"{np.rad2deg(self.cur_angle):.1f} deg, steps: {self.step_count}, "
				f"speed: {self.speed:.2f} m/s"
			)
			if len(text) > 0: self.text_label.text += text
			self.text_label.draw()

		# Force execution of queued commands
		gl.glFlush()

		if self.rec is not None:
			self.rec.capture_frame()

		if self.renderables is not None:
			for r in self.renderables: r.render()

		return img

	def close(self):
		if self.rec is not None:
			self.rec.close()
		super().close()

	def reset(self, segment: bool = False, force = True):
		if force:
			self.force_reset(segment)

	def step(self, pwm_left: float, pwm_right: float):
		self.actions[0], self.actions[1] = pwm_left, pwm_right
		obs, reward, done, info = super().step(self.actions)
		return obs, reward, done, info

	def pointing_direction(self) -> tuple:
		q = math.pi*0.25
		theta = self.cur_angle
		# East
		if q >= theta > -q:
			return 1, 0
		t = math.pi*0.75
		# North
		if t >= theta > q:
			return 0, -1
		# South
		if -q >= theta > -t:
			return 0, 1
		# West
		return -1, 0

	def get_dir_vec(self):
		return gym_duckietown.simulator.get_dir_vec(self.cur_angle)

	def force_reset(self, segment: bool = False):
		super().reset(segment)

	# We have to convert from window positions to actual Duckietown coordinates.
	def convert_coords(self, x: int, y: int) -> (float, float):
		w = self.grid_width+3
		h = self.grid_height+2
		dw = WINDOW_WIDTH/w
		dh = WINDOW_HEIGHT/h
		x -= 1.5*dw
		y += dh
		x, y = x/dw, (WINDOW_HEIGHT-y) / dh
		return (x+0.5)*self.road_tile_size, (y+0.5)*self.road_tile_size

	# The inverse transformation of the above.
	def unconvert_coords(self, x: float, y: float = None) -> (int, int):
		if y is None:
			x, y = x[0], x[1]
		x, y = x/self.road_tile_size-0.5, y/self.road_tile_size-0.5
		w = self.grid_width+3
		h = self.grid_height+2
		dw = WINDOW_WIDTH/w
		dh = WINDOW_HEIGHT/h
		x, y = dw*x, dh*y-WINDOW_HEIGHT
		return int(x + 1.5*dw), int(y - dh)

	def add_duckie(self, x, y = None, static = True, optional = False):
		if y is None:
			x, y = x[0], x[1]
		obj = _get_obj_props('duckie', x, y, static, optional = optional)
		self.objects.append(
			gym_duckietown.objects.DuckieObj(
				obj,
				False,
				gym_duckietown.simulator.SAFETY_RAD_MULT,
				self.road_tile_size
			)
		)

	def add_big_duckie(self, x, y = None, static = True, optional = False):
		if y is None:
			x, y = x[0], x[1]
		obj = _get_obj_props('duckie', x, y, static, rescale = 3.0, optional = optional)
		self.objects.append(
			gym_duckietown.objects.DuckieObj(
				obj, 
				False,
				gym_duckietown.simulator.SAFETY_RAD_MULT,
				self.road_tile_size
			)
		)

	def add_static_duckie(self, x, y = None, angle = None):
		if y is None: x, y = x[0], x[1]
		if angle is None: angle = self.np_random.random()*math.pi
		obj = _get_obj_props('duckie', x, y, True, rescale = 3.0, angle = angle)
		self.objects.append(gym_duckietown.objects.WorldObj(obj, False, gym_duckietown.simulator.SAFETY_RAD_MULT))

	def add_static_big_duckie(self, x, y = None, angle = None):
		if y is None: x, y = x[0], x[1]
		if angle is None: angle = self.np_random.random()*math.pi
		obj = _get_obj_props('duckie', x, y, True, angle = angle)
		self.objects.append(gym_duckietown.objects.WorldObj(obj, False, gym_duckietown.simulator.SAFETY_RAD_MULT))

	def add_static_duckiebot(self, x, y = None, angle = None):
		if y is None: x, y = x[0], x[1]
		if angle is None: angle = self.np_random.random()*math.pi
		obj = _get_obj_props('duckiebot', x, y , True, rescale = 2.0, angle = angle)
		self.objects.append(gym_duckietown.objects.WorldObj(obj, False, gym_duckietown.simulator.SAFETY_RAD_MULT))

	def add_cone(self, x, y = None, scale = 1.0):
		if y is None:
			x, y = x[0], x[1]
		obj = _get_obj_props('cone', x, y, rescale = scale)
		c = gym_duckietown.objects.WorldObj(obj, False, gym_duckietown.simulator.SAFETY_RAD_MULT)
		self.objects.append(c)
		return c

	def add_walking_duckie(self, x, y = None):
		if y is None:
			x, y = x[0], x[1]
		obj = _get_obj_props('duckie', x, y, False)
		obj['kind'] = 'duckiebot'
		d = gym_duckietown.objects.DuckiebotObj(obj, False, gym_duckietown.simulator.SAFETY_RAD_MULT,
											  gym_duckietown.simulator.WHEEL_DIST,
											  gym_duckietown.simulator.ROBOT_WIDTH,
											  gym_duckietown.simulator.ROBOT_LENGTH)
		self.objects.append(d)
		return d

	def add_light(self, x, y):
		li = gl.GL_LIGHT0 + 1

		li_pos = [x, 0.5, y, 1.0]
		diffuse = [0.5, 0.5, 0.5, 0.5]
		ambient = [0.5, 0.5, 0.5, 0.5]
		specular = [0.5, 0.5, 0.5, 1.0]
		spot_direction = [0.5, -0.5, 0.5]
		gl.glLightfv(li, gl.GL_POSITION, (gl.GLfloat * 4)(*li_pos))
		gl.glLightfv(li, gl.GL_AMBIENT, (gl.GLfloat * 4)(*ambient))
		gl.glLightfv(li, gl.GL_DIFFUSE, (gl.GLfloat * 4)(*diffuse))
		gl.glLightfv(li, gl.GL_SPECULAR, (gl.GLfloat * 4)(*specular))
		gl.glLightfv(li, gl.GL_SPOT_DIRECTION, (gl.GLfloat * 3)(*spot_direction))
		# gl.glLightfv(li, gl.GL_SPOT_EXPONENT, (gl.GLfloat * 1)(64.0))
		gl.glLightf(li, gl.GL_SPOT_CUTOFF, 60)

		gl.glLightfv(li, gl.GL_CONSTANT_ATTENUATION, (gl.GLfloat * 1)(1.0))
		# gl.glLightfv(li, gl.GL_LINEAR_ATTENUATION, (gl.GLfloat * 1)(0.1))
		gl.glLightfv(li, gl.GL_QUADRATIC_ATTENUATION, (gl.GLfloat * 1)(0.2))
		gl.glEnable(li)

	def sine_target(self, t: np.ndarray, s: np.ndarray = None):
		if s is None:
			s = np.delete(self.get_dir_vec(), 1)
		u, v = s/np.linalg.norm(s), t/np.linalg.norm(t)
		return np.cross(u, v)/(np.linalg.norm(u, ord=1)*np.linalg.norm(v, ord=1))

	def lf_target(self):
		_, ct, c0 = self.closest_curve_point(self.cur_pos, self.cur_angle, delta = 0.2)
		u, v = np.delete(c0, 1), np.delete(ct, 1)
		p = u-self.get_position()
		t = math.asin(self.sine_target(v))
		d = np.linalg.norm(p)*np.sign(self.sine_target(p, v))
		return d, t

	def tile_position(self, x, y, centered: bool = False) -> (float, float):
		s = self.road_tile_size
		if centered: return x*s+s/2, y*s+s/2
		return x*s, y*s

	road_tiles = set(["curve_left", "curve_right", "straight", "4way", "3way_left", "3way_right"])

	def random_road_pose(self) -> (np.ndarray, float):
		R = [np.array(t["coords"])*self.road_tile_size+self.road_tile_size/2 for t in self.grid if t["kind"] in DuckievillageEnv.road_tiles]
		return np.insert(random.choice(R), 1, 0), math.pi*self.np_random.random()
