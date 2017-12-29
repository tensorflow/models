from __future__ import print_function
import tensorflow as tf
import numpy as np
from collections import deque
from time import sleep
import time
import datetime
import random
import copy
from game import Tetris
from play import TetrisUI
import model as using_model

model = None
sess = None
saver = None
is_new_model = False
is_master = False
fix_learningrate = 0
save_path = ""

def init_model(train = False, forceinit = False, init_with_gold = False, learning_rate = 0):
	global model
	global sess
	global saver
	global is_new_model
	global save_path
	
	# save_path is current save path, and gold_save_path is a stable save for init. it can be a middle version in training
	model = using_model.create_model_5()
	save_path = "model_5/"
	gold_save_path = "model_5_gold/"
	
	with model.as_default():
		global_step = tf.Variable(0, name="step")

	if train:
		create_train_op(model, learning_rate=learning_rate)
	
	sess = tf.Session(graph = model)

	with model.as_default():
		saver = tf.train.Saver(max_to_keep = 1)

		cp = tf.train.latest_checkpoint(save_path)
		if cp == None or forceinit:
			if init_with_gold:
				print("init model with gold val")
				restore_model(sess, gold_save_path)
			else:
				print("init model with default val")
				tf.global_variables_initializer().run(session=sess)
				is_new_model = True
			save_model()
		else:
			print("init model with saved val")
			saver.restore(sess, cp)
			is_new_model = False

def save_model():
	global sess
	global saver
	global save_path
	saver.save(sess, save_path + 'save.ckpt')

def restore_model(dst_sess, src_path = None):
	global saver
	global save_path
	if src_path == None:
		src_path = save_path
	cp = tf.train.latest_checkpoint(src_path)
	if cp != None:
		saver.restore(dst_sess, cp)
	else:
		print("restore model fail.")

__cur_step = -1
__cur_action = 0
def run_game(tetris):
	global model
	global sess
	global __cur_step
	global __cur_action
	if tetris.step() != __cur_step:
		status = train_make_status(tetris)
		__cur_step = tetris.step()
		_, __cur_action = train_getMaxQ(status, model, sess)
		print("step %d, score: %d, action: %d" % (__cur_step, tetris.score(), __cur_action))

	x, r = train_getxr_by_action(__cur_action)
	if tetris.move_step_by_ai(x, r):
		tetris.fast_finish()


def create_train_op(model, learning_rate):
	global fix_learningrate

	with model.as_default():
		#train input
		_action = tf.placeholder(tf.float32, [None, 40], name="action")
		_targetQ = tf.placeholder(tf.float32, [None], name="targetQ") # reward + gamma * max(Q_sa)

		#train
		Q = model.get_tensor_by_name("output:0")		
		cost = tf.reduce_mean(tf.square(Q - _targetQ), name="cost")
		
		global_step = model.get_tensor_by_name("step:0")
		init_lr = 1e-4
		decay_lr = tf.train.exponential_decay(init_lr, global_step, decay_steps=1500, decay_rate=0.98, staircase=True, name="lr")
		if learning_rate == 0:
			optimizer = tf.train.AdamOptimizer(decay_lr).minimize(cost, name="train_op", global_step=global_step)
			fix_learningrate = 0
			print("decay learning rate with init value: %f" % init_lr)
		else:
			optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost, name="train_op", global_step=global_step)
			fix_learningrate = learning_rate
			print("fix learning rate: %f" % learning_rate)

	return model

def train(tetris
	, memory_size = 1000
	, batch_size = 50
	, train_steps = 10000
	, gamma = 0.6
	, init_epsilon = 1
	, min_epsilon = 0.01
	, as_master = False
	, printPerStep = 100
	, upgateTargetAndSavePerStep = 1000
	, ui = None):
	global model
	global sess
	global is_new_model
	global fix_learningrate
	global is_master
	D = deque()

	# Debug
	# batch_size = 1
	# printPerStep = 1
	# Finish Debug

	target_sess = tf.Session(graph = model)
	restore_model(target_sess)
	global_step = sess.run(model.get_tensor_by_name("step:0"))

	# 'master mode' is use for fine turnning, there is a difference of random play and reward function
	is_master = as_master

	if not is_new_model:
		init_epsilon = float(init_epsilon) / 2

	epsilon = init_epsilon
	step = 0
	status_0 = train_make_status(tetris)
	print("train start at: " + time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())) \
			 + ", global step: " + str(global_step) + ", Master: " + str(is_master))
	while True:
		#run game
		action_0 = 0
		if is_master: # in master mode, reduce the random range
			Action_s = train_getActionArrayByQ(status_0, model, sess)
			if random.random() < epsilon:
				selectable_size = int(len(Action_s) / 4)
				action_0 = Action_s[random.randrange(0, selectable_size)]
			else:
				action_0 = Action_s[0]
		else:
			if random.random() < epsilon:
				action_0 = random.randrange(len(train_getValidAction(status_0)))
			else:
				_, action_0 = train_getMaxQ(status_0, model, sess)
		epsilon = init_epsilon + (min_epsilon - init_epsilon) * step / train_steps

		gameover = train_run_game(tetris, action_0, ui)  #use the action to run, then get reward
		if gameover:
			print("train game over, score: %d, step: %d" % (tetris.score(), tetris.step()))
			tetris.reset()
		status_1 = train_make_status(tetris)
		reward_1, reward_info = train_cal_reward(tetris, gameover)

		Q_0 = train_getQ([status_0], [action_0], model, sess)[0]
		targetMaxQ_1, _ = train_getMaxQ(status_1, model, target_sess)
		priproity = abs(Q_0 - reward_1 - targetMaxQ_1) #loss is priproity
		
		#log to memory
		D.append([status_0, action_0, Q_0, reward_1, status_1, targetMaxQ_1, gameover, priproity])
		if len(D) > memory_size:
			D.popleft()

		if ui != None:
			ui.log("reward: %f, info: %s" % (reward_1, reward_info))

		#review memory
		if len(D) > batch_size:
			# batch = random.sample(D, batch_size)
			batch = train_sample(D, batch_size)
			status_0_batch = [d[0] for d in batch]
			action_0_batch = [d[1] for d in batch]
			reward_1_batch = [d[3] for d in batch]
			status_1_batch = [d[4] for d in batch]
			targetMaxQ_1_batch = [d[5] for d in batch]
			gameover_1_batch = [d[6] for d in batch]

			t_caltarget_begin = datetime.datetime.now()
			targetQ_batch = []
			for i in range(len(batch)):
				if gameover_1_batch[i]:
					targetQ_batch.append(reward_1_batch[i])
				else:
					targetQ_batch.append(reward_1_batch[i] + gamma * targetMaxQ_1_batch[i])
			t_caltarget_use = datetime.datetime.now() - t_caltarget_begin

			from_s = []
			to_s = []
			next_s = []
			for i in range(len(status_0_batch)):
				_from, _to, _next = train_simlutate_status_for_model_input(status_0_batch[i], action_0_batch[i])
				from_s.append(_from)
				to_s.append(_to)
				next_s.append(_next)

			t_trainnet_begin = datetime.datetime.now()
			_, _output, _cost, global_step, _lr = sess.run((model.get_operation_by_name("train_op")
				, model.get_tensor_by_name("output:0")
				, model.get_tensor_by_name("cost:0")
				, model.get_tensor_by_name("step:0")
				, model.get_tensor_by_name("lr:0")
				)
				, feed_dict={"from:0":from_s, "to:0":to_s, "next:0":next_s, "targetQ:0":targetQ_batch, "kp:0":0.75})
			t_trainnet_use = datetime.datetime.now() - t_trainnet_begin

			for i in range(len(batch)):
				batch[i][2] = _output[i]
				train_update_sample_rate(batch[i])

			if step % printPerStep == 0:
				match_cnt = 0
				for i in range(batch_size):
					if targetQ_batch[i] != 0 and float(abs(_output[i] - targetQ_batch[i])) / float(abs(targetQ_batch[i])) < 0.1:
						match_cnt += 1
				match_rate = float(match_cnt) / float(batch_size)
				using_lr = fix_learningrate if fix_learningrate > 0 else _lr
				info = "train step %d(g: %d), epsilon: %f, lr: %f, action[0]: %d, reward[0]: %f, targetQ[0]: %f, Q[0]: %f, matchs: %f, cost: %f (time: %d/%d)" \
						% (step, global_step, epsilon, using_lr, action_0_batch[0], reward_1_batch[0], targetQ_batch[0], _output[0], match_rate, _cost \
						, t_caltarget_use.microseconds, t_trainnet_use.microseconds)
				if ui == None:
					print(info)
					if printPerStep == 1:	# for debug
						sleep(1)
				else:
					ui.log(info)

			if step % upgateTargetAndSavePerStep == 0:
				print("update target session...")
				restore_model(target_sess)
				x = 0
				for memory in D:
					memory[5], _ = train_getMaxQ(memory[4], model, target_sess)
					train_update_sample_rate(memory)
					x += 1

				print("save model...")
				save_model()
		#loop
		status_0 = status_1
		step += 1
		if step > train_steps:
			break

	print("train finish at: " + time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

def train_softmax(x):
	return np.exp(x) / np.sum(np.exp(x), axis = 0)

def train_lerp(x0, x1, t):
	return x0 + (x1 - x0) * t

def train_update_sample_rate(m):
	m[7] = abs(m[2] - m[3] - m[5])

def train_sample(D, size):
	# Prioritized Sweeping, study from this:
	# http://cs231n.stanford.edu/reports/2016/pdfs/121_Report.pdf
	priproity = [d[7] for d in D]
	s = float(sum(priproity))
	normal_priproity = [float(p) / s for p in priproity]

	idxs = np.random.choice(len(D), size, False, normal_priproity)
	batch = []
	for i in idxs:
		batch.append(D[i])
	return batch

def train_make_status(tetris):	# 0, tiles; 1, current
	w = tetris.width()
	h = tetris.height()
	image = [[ 0 for x in range(w) ] for y in range(h)]
	
	tiles = tetris.tiles()
	for y in range(0, h):
		for x in range(0, w):
			if tiles[y][x] > 0:
				image[y][x] = 1
	
	_current = tetris.current_index()
	_next = tetris.next_index()
	_score = tetris.score()
	_step = tetris.step()
	status = {"tiles":image, "current":_current, "next":_next, "score":_score, "step":_step}
	return status

def train_getQ(status_s, action_s, use_model, use_sess):
	from_s = []
	to_s = []
	next_s = []
	for i in range(len(status_s)):
		_from, _to, _next = train_simlutate_status_for_model_input(status_s[i], action_s[i])
		from_s.append(_from)
		to_s.append(_to)
		next_s.append(_next)

	Q = use_sess.run(use_model.get_tensor_by_name("output:0"), feed_dict={"from:0":from_s, "to:0":to_s, "next:0":next_s, "kp:0":1.0})
	return Q

def train_getMaxQ(status, use_model, use_sess):
	Q_s = train_getQ_Array(status, use_model, use_sess)
	return max(Q_s), np.argmax(Q_s)

def train_getQ_Array(status, use_model, use_sess):
	status_s = []
	action_s = []
	for i in train_getValidAction(status):
		status_s.append(status)
		action_s.append(i)
	Q_s = train_getQ(status_s, action_s, use_model, use_sess)
	return Q_s

def train_getActionArrayByQ(status, use_model, use_sess):
	Q_s = train_getQ_Array(status, use_model, use_sess)
	Action_s = []
	for i in range(len(Q_s)):
		a = np.argmax(Q_s)
		Q_s[a] = -10000
		Action_s.append(a)
	return Action_s

def train_getMaxQ_batch(status_batch, use_model, use_sess):
	status_s = []
	action_s = []
	for status in status_batch:
		for i in train_getValidAction(status):
			status_s.append(status)
			action_s.append(i)
	Q_s = train_getQ(status_s, action_s, use_model, use_sess)

	maxQ_batch = []
	maxAction_batch = []
	p = 0
	for status in status_batch:
		actLen = len(train_getValidAction(status))
		smallQ_s = Q_s[p:p+actLen]
		p += actLen
		maxQ_batch.append(max(smallQ_s))
		maxAction_batch.append(np.argmax(smallQ_s))

	return maxQ_batch, maxAction_batch

def train_getValidAction(status):
	current = status["current"]
	if current == 1:
		return range(10)
	elif current == 0 or current == 3 or current == 4:
		return range(20)
	else:
		return range(40)

_simulator = Tetris()
def train_simlutate_status_for_model_input(status, action):
	global _simulator
	_simulator.apply_status_by_ai(nodes = status["tiles"], _current = status["current"], _next = status["next"], _score = status["score"], _step = status["step"])
	image_from = train_capture_model_input_image(_simulator)
	next_index = _simulator.next_index()
	train_run_game(_simulator, action, None)
	image_to = train_capture_model_input_image(_simulator)
	return image_from, image_to, next_index

def train_capture_model_input_image(tetris):
	w = tetris.width()
	h = tetris.height()
	image = [[ 0 for x in range(w)] for y in range(h)]
	
	tiles = tetris.tiles()
	for y in range(0, h):
		for x in range(0, w):
			if tiles[y][x] > 0:
				image[y][x] = 1

	current = tetris.current()
	for t in current:
		image[t[1]][t[0]] = 1
	return image

def train_getxr_by_action(action):
	r = int(action / 10)
	x = int(action % 10)
	return x, r

def train_run_game(tetris, action, ui):
	x, r = train_getxr_by_action(action)

	while True:
		move_finish = tetris.move_step_by_ai(x, r)

		if ui != None:
			if ui.refresh_and_check_quit():
				raise Exception("user quit")

		if move_finish:
			tetris.fast_finish()
			break

	return tetris.gameover()

s_last_aggregate_height = 0
s_last_holes = 0
s_last_bumpiness = 0

def train_reset_reward_status():
	global s_last_aggregate_height
	global s_last_holes
	global s_last_bumpiness
	s_last_aggregate_height = 0
	s_last_holes = 0
	s_last_bumpiness = 0

def train_heuristic_score(aggregate_height, complete_lines, holes, bumpiness):
	score = -0.510066 * aggregate_height + 0.760666 * complete_lines - 0.35663 * holes - 0.184483 * bumpiness
	return score

def train_cal_reward(tetris, gameover = False):
	# study from this:
	# https://codemyroad.wordpress.com/2013/04/14/tetris-ai-the-near-perfect-player/
	global s_last_aggregate_height
	global s_last_holes
	global s_last_bumpiness
	global is_master

	if gameover:
		train_reset_reward_status()
		return -100, "game over"

	complete_lines = tetris.last_erase_row()

	column_height = [0] * 10
	holes = 0

	tiles = tetris.tiles()
	for y in range(len(tiles)):
		height = 20-y
		row = tiles[y]
		for x in range(len(row)):
			t = row[x]
			if t > 0:
				column_height[x] = max(column_height[x], height)
			elif height < column_height[x]:
				holes += 1

	aggregate_height = sum(column_height)
	aggregate_height_just_clear = complete_lines * 10
	bumpiness = sum([abs(column_height[i] - column_height[i+1]) for i in range(9)])

	reward = train_heuristic_score(aggregate_height + aggregate_height_just_clear - s_last_aggregate_height
		, complete_lines
		, holes - s_last_holes
		, bumpiness - s_last_bumpiness)

	info = "aggregate_height: %d, complete_lines: %d, holes: %d, bumpiness: %d" % (aggregate_height, complete_lines, holes, bumpiness)

	s_last_holes = holes
	s_last_bumpiness = bumpiness

	if is_master:
		reward += complete_lines * complete_lines

	return reward, info

if __name__ == '__main__':
	init_model()
	save_model()
