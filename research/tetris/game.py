import random
import time

# shapes
group_table = [
	[[[0,-1],[0,0],[0,1],[0,2]], [[-1,0],[0,0],[1,0],[2,0]]],
	[[[0,0],[0,1],[1,0],[1,1]]],
	[[[0,0],[-1,0],[0,-1],[1,0]], [[0,0],[0,1],[0,-1],[1,0]], [[0,0],[-1,0],[0,1],[1,0]], [[0,0],[-1,0],[0,-1],[0,1]]],
	[[[-1,-1],[0,-1],[0,0],[1,0]], [[1,-1],[1,0],[0,0],[0,1]]],
	[[[-1,0],[0,0],[0,-1],[1,-1]], [[-1,-1],[-1,0],[0,0],[0,1]]],
	[[[0,1],[0,0],[0,-1],[1,1]], [[0,0],[1,0],[-1,0],[-1,1]], [[-1,-1],[0,-1],[0,0],[0,1]], [[0,0],[-1,0],[1,0],[1,-1]]],
	[[[0,1],[0,0],[0,-1],[-1,1]], [[0,0],[1,0],[-1,0],[-1,-1]], [[1,-1],[0,-1],[0,0],[0,1]], [[0,0],[-1,0],[1,0],[1,1]]],
]

score_table = [0, 1, 2, 4, 6]

class Tetris:
	__width = 10
	__height = 20
	__tiles = None
	__curIdx = 0
	__curVal = 0
	__curX = 0
	__curY = 0
	__curR = 0
	__nextIdx = 0
	__nextVal = 0
	__score = 0
	__step = 0
	__last_earse_row = 0
	__gameover = False

	__dbginfo = ""

	def width(self):
		return self.__width

	def height(self):
		return self.__height

	def tiles(self):
		return self.__tiles

	def current(self):
		return self.__gen_shape(self.__curIdx, self.__curVal, self.__curX, self.__curY, self.__curR)

	def next(self):
		return self.__gen_shape(self.__nextIdx, self.__nextVal, 1, 1, 0)

	def current_index(self):
		return self.__curIdx

	def current_X(self):
		return self.__curX

	def current_Y(self):
		return self.__curY

	def current_rotate(self):
		return self.__curR

	def next_index(self):
		return self.__nextIdx

	def score(self):
		return self.__score

	def step(self):
		return self.__step

	def last_erase_row(self):
		return self.__last_earse_row

	def gameover(self):
		return self.__gameover

	def dbginfo(self):
		return self.__dbginfo

	def clear_dbginfo(self):
		self.__dbginfo = ""

	def move_current(self, x = 0, y = 0):
		tmp = self.__gen_shape(self.__curIdx, self.__curVal, self.__curX + x, self.__curY + y, self.__curR)
		if self.__test_collision(tmp):
			if y != 0:
				self.__finish_current()
				self.__pop_next()
			return
		self.__curX += x
		self.__curY += y

	def rotate_current(self):
		newR = (self.__curR + 1) % len(group_table[self.__curIdx])
		if not self.__test_collision(self.__gen_shape(self.__curIdx, self.__curVal, self.__curX, self.__curY, newR)):
			self.__curR = newR
			return True
		if not self.__test_collision(self.__gen_shape(self.__curIdx, self.__curVal, self.__curX + 1, self.__curY, newR)):
			self.__curR = newR
			self.__curX += 1
			return True
		if not self.__test_collision(self.__gen_shape(self.__curIdx, self.__curVal, self.__curX - 1, self.__curY, newR)):
			self.__curR = newR
			self.__curX -= 1
			return True
		if not self.__test_collision(self.__gen_shape(self.__curIdx, self.__curVal, self.__curX + 2, self.__curY, newR)):
			self.__curR = newR
			self.__curX += 2
			return True
		if not self.__test_collision(self.__gen_shape(self.__curIdx, self.__curVal, self.__curX - 2, self.__curY, newR)):
			self.__curR = newR
			self.__curX -= 2
			return True
		return False

	def fast_finish(self):
		step = self.__step
		while self.__step == step:
			self.move_current(y = 1)

	def move_step_by_ai(self, x, rotate):
		l = len(group_table[self.__curIdx])
		if self.__curR % l != rotate % l:
			if self.rotate_current():
				return False

		oldX = self.__curX
		if self.__curX > x:
			self.move_current(x = -1)
		elif self.__curX < x:
			self.move_current(x = 1)
		
		if oldX != self.__curX:
			return False

		return True

	def apply_status_by_ai(self, nodes, _current, _next, _score, _step):
		for y in range(0, self.height()):
			for x in range(0, self.width()):
				self.__tiles[y][x] = nodes[y][x]
		self.__curIdx = _current
		self.__nextIdx = _next
		self.__score = _score
		self.__step = _step
		
		self.__last_earse_row = 0
		self.__gameover = False
		self.__curVal = 1
		self.__curX = self.width() / 2
		self.__curY = 1
		self.__curR = 0


	def reset(self):
		self.__tiles = [[ 0 for x in range(self.__width) ] for y in range(self.__height)]
		self.__curIdx = 0
		self.__curVal = 0
		self.__curX = 0
		self.__curY = 0
		self.__curR = 0
		self.__nextIdx = 0
		self.__nextVal = 0
		self.__score = 0
		self.__step = 0
		self.__last_earse_row = 0
		self.__gameover = False
		self.__dbginfo = ""
		
		self.__gen_next()
		self.__pop_next()

	def random_tiles(self, h, r = 0.8):
		for y in range(20-h, 20):
			for x in range(10):
				if random.random() < r:
					self.__tiles[y][x] = random.randint(1, 9)

	def __init__(self):
		random.seed(time.time())
		self.reset()

	def __gen_next(self):
		self.__nextIdx = random.randint(0, len(group_table)-1)
		self.__nextVal = random.randint(1, 9)

	def __pop_next(self):
		self.__curIdx = self.__nextIdx
		self.__curVal = self.__nextVal
		self.__curX = self.width() / 2
		self.__curY = 1
		self.__curR = 0
		self.__step = self.__step + 1
		self.__gen_next()

		if self.__test_collision(self.current()):
			self.__gameover = True

	def __gen_shape(self, idx, val, cx, cy, r):
		return [[cx + t[0], cy + t[1], val] for t in group_table[idx][r]];

	def __test_collision(self, group):
		for t in group:
			x = t[0]
			y = t[1]
			if x < 0 or x > self.width() - 1 or y < 0 or y > self.height() - 1:
				return True
			if self.__tiles[y][x] > 0:
				return True
		return False

	def __finish_current(self):
		for t in self.current():
			x = t[0]
			y = t[1]
			v = t[2]
			self.__tiles[y][x] = v

		row_cnt = 0
		for row in range(0, self.__height):
			if self.__is_full_row(row):
				self.__clear_row(row)
				row_cnt += 1
		self.__score += score_table[row_cnt]
		self.__last_earse_row = row_cnt

	def __is_full_row(self, row):
		for x in range(self.__width):
			if(self.__tiles[row][x] == 0):
				return False
		self.__dbginfo = "full row: " + str(row)
		return True

	def __clear_row(self, row):
		for i in reversed(range(0, row)):
			for x in range(self.__width):
				self.__tiles[i + 1][x] = self.__tiles[i][x]
		for x in range(self.__width):
			self.__tiles[0][x] = 0

if __name__ == '__main__':
	game = Tetris()
	print(game.tiles())
	print(game.next())
	print(game.current())
	game.move_current(y = 1)
	print(game.current())
	del game
