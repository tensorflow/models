import curses
import sys
import getopt
import robot as robot
from game import Tetris
from collections import deque
from time import sleep

key_up = 65
key_down = 66
key_left = 68
key_right = 67
key_space = 32

class TetrisUI:
	__baseX = 4
	__baseY = 2
	__tileWidth = 2
	__tetris = None
	__scr = None
	__lastkey = 0
	__train_info = None
	__infoview_y = 0
	__log_info = None

	def __init__(self, tetris, frame_interval = 500):
		print("init tetris gui")
		self.__tetris = tetris
		self.__log_info = deque()
		self.__scr = curses.initscr()
		curses.noecho()
		curses.cbreak()
		self.__scr.timeout(frame_interval)

	def __del__(self):
		self.__tetris = None
		self.__scr = None
		curses.nocbreak()
		curses.echo()
		curses.endwin()
		print("destory tetris gui")

	def __refresh(self):
		self.__scr.clear()
		self.__scr.border()
		tiles = self.__tetris.tiles()
		width = self.__tetris.width()
		height = self.__tetris.height()

		for y in range(0, height):
			for x in range(0, width):
				tile = tiles[y][x]
				self.__drawtile(x, y, tile)

		current = self.__tetris.current()
		for t in current:
			self.__drawtile(t[0], t[1], t[2])

		info_x = (self.__tetris.width() + 3) * self.__tileWidth
		info_y = self.__baseY + 8

		_next = self.__tetris.next()
		for t in _next:
			self.__drawtile(t[0], t[1], t[2], baseX = info_x)

		self.__reset_infoview()
		self.__drawinfoview("INFO")
		self.__drawinfoview("Score: %d" % self.__tetris.score())
		self.__drawinfoview("Step: %d" % self.__tetris.step())
		self.__drawinfoview("LastErase: %d" % self.__tetris.last_erase_row())
		self.__drawinfoview("LastKey: %d" % self.__lastkey)
		self.__drawinfoview("Dbg: %s" % self.__tetris.dbginfo())
		if self.__train_info != None:
			self.__drawinfoview(self.__train_info)
		if self.__tetris.gameover():
			self.__drawinfoview("GAME OVER")

		self.__drawlogview()


	def __drawtile(self, x, y, v, baseX = 0, baseY = 0):
		ch = '.'
		if v != 0:
			ch = str(v)
		if baseX == 0:
			baseX = self.__baseX
		if baseY == 0:
			baseY = self.__baseY
		self.__drawcontent(baseX + x * self.__tileWidth, baseY + y, ch)

	def __drawcontent(self, x, y, s):
		self.__scr.addstr(y, x, s)

	def __reset_infoview(self):
		self.__infoview_y = self.__baseY + 8

	def __drawinfoview(self, s):
		info_x = (self.__tetris.width() + 3) * self.__tileWidth
		info_y = self.__infoview_y
		self.__drawcontent(info_x, info_y, s)
		self.__infoview_y += 1

	def __drawlogview(self):
		info_x = (self.__tetris.width() + 3) * self.__tileWidth + 25
		info_y = self.__baseY
		for info in self.__log_info:
			self.__drawcontent(info_x, info_y, info)
			info_y += 1

	def loop(self, ai_model = None):
		while True:
			self.__refresh()
			self.__tetris.clear_dbginfo()
			c = self.__scr.getch()

			if self.__tetris.gameover():
				if c == ord('q'):
					print("exit")
					break
				else:
					continue

			if c < 0:
				if ai_model == None:
					self.__tetris.move_current(y = 1)
				else:
					ai_model.run_game(self.__tetris)
			elif c == key_left:
				if ai_model == None:
					self.__tetris.move_current(x = -1)
			elif c == key_right:
				if ai_model == None:
					self.__tetris.move_current(x = 1)
			elif c == key_down:
				if ai_model == None:
					self.__tetris.move_current(y = 1)
			elif c == key_up:
				if ai_model == None:
					self.__tetris.rotate_current()
			elif c == key_space:
				if ai_model == None:
					self.__tetris.fast_finish()
			elif c == ord('q'):
				print("exit")
				break
			
			if c > 0:
				self.__lastkey = c

	def refresh_and_check_quit(self):
		self.__refresh()
		c = self.__scr.getch()
		return c == ord('q')

	def log(self, info):
		self.__log_info.append(info)
		if len(self.__log_info) > 30:
			self.__log_info.popleft()
			
def play():
	game = Tetris()
	ui = TetrisUI(game)
	ui.loop()
	del ui
	del game

def play_train(with_ui = False, force_init = False, init_with_gold = False, train_count = 0, learn_rate = 0, is_master = False, ui_tick = 0):
	robot.init_model(train = True, forceinit = force_init, init_with_gold = init_with_gold, learning_rate = learn_rate)
	game = Tetris()
	ui = None
	if with_ui:
		if ui_tick == 0:
			ui_tick = 100
		ui = TetrisUI(game, ui_tick)
	try:
		if train_count == 0:
			robot.train(game, as_master = is_master, ui = ui)
		else:
			robot.train(game, train_steps = train_count, as_master = is_master, ui = ui)
	except KeyboardInterrupt:
		print("user exit")
	robot.save_model()
	if ui != None:
		del ui
	del game

def play_ai():
	game = Tetris()
	robot.init_model()
	ui = TetrisUI(game, 250)
	try:
		ui.loop(ai_model = robot)
	except KeyboardInterrupt:
		print("user exit")
	del ui
	del game

def play_ai_without_ui(count):
	if count == 0:
		count = 10
	game = Tetris()
	robot.init_model()
	scores = []
	for i in range(count):
		while not game.gameover():
			robot.run_game(game)
			# sleep(0.5)
		scores.append(game.score())
		print("game over, score: %d, step: %d" % (game.score(), game.step()))
		game.reset()
	del game

	print("max: %d, min: %d, avg: %d" % (max(scores), min(scores), float(sum(scores)) / len(scores)))

def print_help():
	print("help:")
	print("use python2")
	print("play.py")
	print("    play your self")
	print("play.py -a")
	print("    use ai to play game")
	print("play.py -A0")
	print("    use ai to play game, without ui")
	print("    -Ax, x is play times, 0 for 10")
	print("play.py -t0 [-n [-g]] [-m] [-l0] [-u0]")
	print("    train model")
	print("    -tx, x is train times, 0 for 10000")
	print("    -n, create a new model to train")
	print("    -g, use with -n, create a new model from golden version, which saved in golden path")
	print("    -m, use master train mode, which have difference random action strategies and reward function")
	print("    -lx, specify learn rate, 0 for default decay")
	print("    -ux, train with ui, use for test, x is ui frame interval in ms")

if __name__ == '__main__':
	mode = "play"
	train_with_ui = False
	train_force_init = False
	train_init_with_gold = False
	train_count = 0
	train_learnrate = 0
	train_ismaster = False
	ai_dbg_count = 0
	ui_tick = 0
	opts, _ = getopt.getopt(sys.argv[1:], "t:aA:u:ngl:mh")
	for op, value in opts:
		if op == "-t":
			mode = "train"
			train_count = int(value)
		elif op == "-a":
			mode = "ai"
		elif op == "-A":
			mode = "ai_dbg"
			ai_dbg_count = int(value)
		elif op == "-u":
			train_with_ui = True
			ui_tick = int(value)
		elif op == "-n":
			train_force_init = True
		elif op == "-g":
			train_init_with_gold = True
		elif op == "-l":
			train_learnrate = float(value)
		elif op == "-m":
			train_ismaster = True
		elif op == "-h":
			mode = "help"
			print_help()

	if mode == "play":
		play()
	elif mode == "train":
		play_train(with_ui=train_with_ui, force_init=train_force_init, init_with_gold = train_init_with_gold
			, train_count=train_count, learn_rate = train_learnrate, is_master = train_ismaster, ui_tick=ui_tick)
	elif mode == "ai":
		play_ai()
	elif mode == "ai_dbg":
		play_ai_without_ui(ai_dbg_count)
