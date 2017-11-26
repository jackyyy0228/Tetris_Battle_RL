<<<<<<< HEAD
#basic modules needed for game to run
import numpy as np
import random
import pygame
import copy
from move import ipieces,opieces,jpieces,lpieces,zpieces,spieces,tpieces,zeropieces,allpieces,allpieces_name
from screen import Screen
#TODO
#Tspin check
GAME_OVER_REWARD = -0.2

class Piece: #denote the piece we can move
    def __init__(self,b,block,px = 4,py = -2,is_final = False):
        self.b = b
        self.block = block
        self.px = px
        self.py = py
        self.is_final = is_final
    def piece_type(self):
        for idx,piece in enumerate(allpieces):
            if piece == self.b:
                return allpieces_name[idx] 
    def _index(self):
        return self.b.index(self.block)
    def rotate(self):
        c = self._index()
        y = (c+1) % 4
        self.block = self.b[y]
    def counter_rotate(self):
        c = self._index()
        y = (c+3) % 4
        self.block = self.b[y]
    def get_state(self): # for bfs
        return (self.px,self.py,self._index())


class Tetris:
    def __init__(self,action_type = 'grouped',use_fitness = False,is_display = False,_screen = None):
        self.is_display = is_display
        self.action_type = action_type
        self.use_fitness = use_fitness
        if self.is_display:
            if _screen:
                self.screen = _screen
            else:
                self.screen = Screen()
    def reset(self):
        self.grid = [[0]*20 for i in range(10)]
        self.nextlist = []
        self.piecelist=[ipieces,opieces,jpieces,
                         lpieces,zpieces,spieces,tpieces]
        for i in range (5):
            self.nextlist.append(random.choice(self.piecelist))
            self.piecelist.remove(self.nextlist[-1])
        self.piece = None
        self.newBlock()
        self.combo = 0
        self.b2b = False # back to back tetris/tspin
        self.held = zeropieces
        self.change = False
        self.done = False
        self.prev_action = None
        self.positions=[] #block position, used for drawing screen
        self.totalSent = 0
        self.totalCleared = 0
        self.step_cnt = 0
        if self.is_display:
            self.display(reset=True)
        return self.get_state(self.grid)
    def step(self,action):
        if self.check_collide(self.piece):
            self.done = True
        if self.done:
            return self.get_state(),np.full([53],GAME_OVER_REWARD),self.done,0,0
        self.step_cnt += 1
        if self.action_type == 'grouped':
            return self.groupedAction(action)
        elif self.action_type == 'single': #haven't finished
            return self.singleAction(action)
    def groupedAction(self,actionID):
        #actionID = 0~4*13+1(hold) = 53
        #rotate 0~3 times, left 0~6 times right 0~6 times
        if self.use_fitness :
            reward_fitness = self.cal_fitness_reward()
        line_sent = 0
        line_cleared = 0
        if actionID == 52:
            if not self.change:
                self.change = True
                self.hold()
        else:
            self.change = False
            rotate_num = actionID % 4
            line = actionID // 4 - 6
            for i in range(rotate_num):
                self.rotate()
            if line < 0:
                leftTime = -line
                for i in range(leftTime):
                    self.piece = self.moveLeft(self.piece)
            elif line > 0:
                rightTime = line
                for i in range(rightTime):
                    self.piece = self.moveRight(self.piece)
            self.piece = self.hardDrop(self.piece)
            line_sent,line_cleared = self.check_end()
            if not self.done:
                self.newBlock()
        if self.is_display :
            self.display()
        if self.use_fitness :
            return self.get_state(self.grid),reward_fitness,self.done,line_sent,line_cleared
        reward = line_sent
        return self.get_state(self.grid),reward,self.done,line_sent,line_cleared
    def singleAction(self,actionID):
        #actionlist = ['Right','Left','Down','Rotate','counter_rotate','HardDrop','Hold']
        line_cleared = 0
        if actionID == 0:
            self.piece = self.moveRight(self.piece)
        elif actionID == 1:
            self.piece = self.moveLeft(self.piece)
        elif actionID == 2:
            self.piece = self.moveDown(self.piece)
        elif actionID == 3:
            self.piece = self.rotate(self.piece)
        elif actionID == 4:
            self.piece = self.counterRotate(self.piece)
        elif actionID == 5:
            self.piece = self.hardDrop(self.piece)
        elif actionID == 6:
            if not self.change:
                self.change = True
                self.hold()
        reward = 0
        if self.piece.is_final:
            self.change = False
            reward,line_cleared = self.check_end()
            if not self.done:
                self.newBlock()
        if self.is_display:
            self.display()
        self.prev_action = actionID
        return self.get_state(self.grid),reward,self.done,reward,line_cleared
    def check_collide(self,piece):
        for x in range(4):
            for y in range(4):
                if piece.block[x][y] > 0:
                    if not 10 > piece.px + x >= 0:
                        return True
                    if piece.py + y >= 20:
                        return True
                    if piece.py + y >= 0 and self.grid[piece.px + x][piece.py + y] > 0:
                        return True
        return False
    def moveRight(self,piece):
        new_piece = copy.copy(piece)
        new_piece.px = piece.px + 1
        if not self.check_collide(new_piece):
            return new_piece
        return piece
    def moveLeft(self,piece):
        new_piece = copy.copy(piece)
        new_piece.px = piece.px - 1
        if not self.check_collide(new_piece):
            return new_piece
        return piece
    def moveDown(self,piece):
        piece = copy.copy(piece)
        new_piece = copy.copy(piece)
        new_piece.py = piece.py + 1
        if not self.check_collide(new_piece):
            return new_piece
        piece.is_final = True
        return piece
    def hardDrop(self,piece):
        piece = copy.copy(piece)
        while not piece.is_final:
            piece = self.moveDown(piece)
        return piece
    def hold(self):
        if self.held != zeropieces :
            b = self.held
            self.held = self.piece.b
            self.piece = Piece(b,b[0],4,-2)
        else:
            self.held = self.piece.b
            self.newBlock()
    def rotate(self,piece): #0,2橫
        ori_piece = copy.copy(piece)
        piece = copy.copy(piece)
        piece.rotate()
        piece_type = piece.piece_type()
        piece_index = piece._index()
        check_list = [(0,0),(1,0),(-1,0)]
        if piece_type == 'I':
            check_list += [(2,0),(-2,0)]
        for px,py in check_list:
            new_piece = copy.copy(piece)
            new_piece.px,new_piece.py =  new_piece.px + px , new_piece.py + py
            if not self.check_collide(new_piece):
                return new_piece
        check_list = [(-1,2),(-2,1),(1,2),(2,1),(-1,-2),(-2,-1),(1,-2),(2,-1)]
        if piece_type in ['J','L','Z','S'] and piece_index in [0,2]:
            check_list = [(-1,1),(1,1),(-1,-1),(1,-1)] + check_list
        if piece_type == 'T':
            check_list = [(-1,1),(1,1),(-1,-1),(1,-1)] + check_list
        for px,py in check_list:
            new_piece = copy.copy(piece)
            new_piece.px,new_piece.py =  new_piece.px + px , new_piece.py + py
            if not self.check_collide(new_piece):
                return new_piece
        return ori_piece
    def counterRotate(self,piece):
        ori_piece = copy.copy(piece)
        piece = copy.copy(piece)
        piece.counter_rotate()
        piece_type = piece.piece_type()
        piece_index = piece._index()
        check_list = [(0,0),(1,0),(-1,0)]
        if piece_type == 'I':
            check_list += [(2,0),(-2,0)]
        for px,py in check_list:
            new_piece = copy.copy(piece)
            new_piece.px,new_piece.py =  new_piece.px + px , new_piece.py + py
            if not self.check_collide(new_piece):
                return new_piece
        check_list = [(-1,2),(-2,1),(1,2),(2,1),(-1,-2),(-2,-1),(1,-2),(2,-1)]
        if piece_type in ['J','L','Z','S'] and piece_index in [0,2]:
            check_list = [(-1,1),(1,1),(-1,-1),(1,-1)] + check_list
        if piece_type == 'T':
            check_list = [(-1,1),(1,1),(-1,-1),(1,-1)] + check_list
        for px,py in check_list:
            new_piece = copy.copy(piece)
            new_piece.px,new_piece.py =  new_piece.px - px , new_piece.py + py # minus
            if not self.check_collide(new_piece):
                return new_piece
        return ori_piece
        
        return
    def check_tspin(self,piece):
        piece_type = piece.piece_type()
        piece_index = piece._index()
        tspin_type = None
        if piece_type == 'T' and self.prev_action in [3,4]:
            n_grid = 0
            for x,y in [(0,1),(0,3),(2,1),(2,3)]:
                x,y = piece.px + x,piece.py +y
                if not 0 <= x <= 9:
                    n_grid += 1
                elif y >= 20:
                    n_grid += 1
                elif y >= 0 and self.grid[x][y] > 0:
                    n_grid += 1
            if n_grid >= 3:
                tspin_type = 'standard'
                x1,y1 = piece.px + 3,piece.py + 2
                x2,y2 = piece.px + 2,piece.py + 1
                if piece_index == 1:
                    if 0 <= x <= 9  and self.grid[x1][y1] == 0 and self.grid[x2][y2] == 0:
                        tspin_type = 'mini'
                x1,y1 = piece.px - 1,piece.py + 2
                x2,y2 = piece.px + 0,piece.py + 1
                if piece_index == 3:
                    if 0 <= x <= 9  and self.grid[x1][y1] == 0 and self.grid[x2][y2] == 0:
                        tspin_type = 'mini'
                if piece_index == 0:
                    tspin_type = 'mini'
        return tspin_type
    def newBlock(self):
        if len(self.piecelist) == 0:
            self.piecelist = [ipieces,opieces,jpieces,lpieces,zpieces,spieces,tpieces]
        n=random.randint(0,len(self.piecelist)-1) 
        self.nextlist.append(self.piecelist[n])
        self.piecelist.remove(self.piecelist[n])
        b = self.nextlist[0]
        self.piece = Piece(b,b[0],4,-2)
        self.nextlist.remove(self.nextlist[0])
    def check_end(self): # return the line sent
        ##check tspin
        tspin = self.check_tspin(self.piece)
        self.grid = self.place_piece(self.grid,self.piece)
        ## check line cleared
        total_clear = 0
        self.grid,total_clear = cal_lines(self.grid)
        sent = 0
        is_special = False
        if total_clear > 0:
            if self.combo > 8:
                sent = total_clear -1 + 4
            else:
                sent = total_clear -1 + (self.combo+1) // 2
            if total_clear == 4:
                if self.b2b:
                    sent += 2
                else:
                    sent += 1
                is_special = True
            if tspin == 'mini':
                sent += 1
                is_special = True
            if tspin == 'standard':
                for clear_lines in [1,2,3]:
                    if total_clear == clear_lines:
                        sent += clear_lines + 1
                        if self.b2b:
                            sent += clear_lines
                is_special = True
        if total_clear > 0:
            self.combo += 1
        else:
            self.combo = 0
        self.b2b = is_special
        self.totalSent += sent
        self.totalCleared += total_clear
        ## check end game
        for x in range (4):
            for y in range (4):
                if self.piece.block[x][y] > 0:
                    if self.piece.py+y < 0:
                        self.done = True
                        return 0,0
        return sent,total_clear
    def cal_fitness_reward(self):
        rewards=[]
        for actionID in range(53):
            save_item = self._save()
            if actionID != 52:
                self.change = False
                rotate_num = actionID % 4
                line = actionID // 4 - 6
                for i in range(rotate_num):
                    self.rotate()
                if line < 0:
                    leftTime = -line
                    for i in range(leftTime):
                        self.moveLeft()
                elif line > 0:
                    rightTime = line
                    for i in range(rightTime):
                        self.moveRight()
                self.hardDrop()
                self.grid = self.place_piece(self.grid,self.piece)
                reward = self.cal_fitness(self.grid)
                rewards.append(reward)
            self._load(save_item)
        rewards = np.array(rewards)
        if rewards.std() == 0:
            normed = rewards
        else:
            normed = (rewards - rewards.mean()) / rewards.std()
        normed = np.append(normed,0)
        return normed
        
    def cal_fitness(self,grid,lines = -1):
        if lines == -1 :
            grid,lines = cal_lines(grid)
        height = cal_height(grid)
        bumpiness = cal_bumpiness(grid)
        holes = cal_holes(grid)
        return -0.51 * height + 0.76 * lines - 0.36 * holes - 0.18 * bumpiness 
    def getPositions(self,piece):  
        positions=[]
        for x in range(4):
            for y in range(4):
                if piece.block[x][y] > 0:
                    positions.append((piece.px+x,piece.py+y))
                    sorted(positions, key=lambda pos: pos[1])
        return positions
    def place_piece(self,grid,piece):
        temp_grid = copy.deepcopy(grid)
        for x in range(4):
            for y in range(4):
                if piece.block[x][y] > 0:
                    if 10 > piece.px + x >= 0 and 20 > piece.py + y >=0:
                        temp_grid[piece.px + x][piece.py+y] = piece.block[x][y]
        return temp_grid
    def get_state(self,grid): #[grid,nextlist[0]....,held]
        statelist=[]
        #temp_grid = self.place_piece(self.grid,self.piece)
        height = cal_height(grid)
        bumpiness = cal_bumpiness(grid)
        holes = cal_holes(grid)
        temp_grid = np.array(grid)
        grid = np.where( temp_grid > 0 ,1,0)
        grid = np.reshape(grid.T,(20,10,1))
        templist = list(self.nextlist)
        templist.insert(0,self.piece.b)
        for idx,piece in enumerate(templist):
            num = allpieces.index(piece)
            X = np.zeros(len(allpieces) + 1)
            X[num] = 1
            if idx > 0:
                Y = np.concatenate((Y,X))
            else:
                Y = np.array(X)
        Y = np.concatenate((Y,(height,bumpiness,holes))) # 45dim
        return [grid,Y]
    def display(self,reset=False):
        positions = self.getPositions(self.piece)
        hd_piece = self.hardDrop(self.piece)
        self.screen.drawScreen(self.grid,self.piece.px,self.piece.py,hd_piece.px,
                               hd_piece.py,self.piece.block,self.held,self.nextlist,
                               positions,self.totalSent,self.step_cnt,reset=reset)
    def all_possible_state(self): #bfs all possible move
        record = self.bfs(self.piece)
        if self.held != zeropieces :
            b = self.held
            piece = Piece(b,b[0],4,-2)
            record_held = self.bfs(piece)
            record = record + record_held 
        else:
            b = self.nextlist[0]
            piece = Piece(b,b[0],4,-2)
            record_held = self.bfs(piece)
            record = record + record_held
        state_list = []
        for piece in record:
            temp_grid = self.place_piece(self.grid,piece)
            state_list.append(self.get_state(temp_grid))
        return state_list
    def test(self):
        state_list = self.all_possible_state()
        print(self.grid)
        for grid,Y in state_list:
            for i in range(20):
                for j in range(10):
                    self.grid[j][i] = grid[i][j][0]
                    print(grid[i][j][0],end='')
                print()
            print(self.grid)
            #self.display()
            import time
            time.sleep(1)

    def bfs(self,piece):
        start = piece.get_state() #(px,py,rotate_index)
        visited = set()
        queue = [start]
        record = []
        idx = 0
        while idx < len(queue):
            now_state = queue[idx]
            idx += 1
            if now_state in visited:
                continue
            now_piece = Piece(piece.b,piece.b[now_state[2]],now_state[0],now_state[1])
            visited.add(now_state)
            for actionID in range(6):
                if actionID == 0:
                    new_piece = self.moveRight(now_piece)
                elif actionID == 1:
                    new_piece = self.moveLeft(now_piece)
                elif actionID == 2:
                    new_piece = self.moveDown(now_piece)
                elif actionID == 3:
                    new_piece = self.rotate(now_piece)
                elif actionID == 4:
                    new_piece = self.counterRotate(now_piece)
                elif actionID == 5:
                    new_piece = self.hardDrop(now_piece)
                next_state = new_piece.get_state()
                if new_piece.is_final and next_state not in visited:
                    record.append(new_piece)
                    visited.add(next_state)
                    continue
                queue.append(next_state)
        return record                

def cal_lines(grid):
    total_clear = 0
    for y in range(20):
        is_clear = True
        for x in range(10):
            if not 8 > grid[x][y] > 0:
                is_clear = False
        if is_clear:
            total_clear += 1
            for x in range(10):
                for y2 in range(y,0,-1):
                    grid[x][y2] = grid[x][y2-1]
                grid[x][0] = 0
    return grid,total_clear
def cal_holes(grid):
    num_holes = 0
    for x in range(10):
        highest = 20
        for y in range(20):
            if grid[x][y] > 0 and y < highest:
                highest = y
            if grid[x][y] == 0 and y > highest:
                num_holes += 1
    return num_holes
def cal_height(grid):
    sum_height = 0
    for x in range(10):
        height = 0
        for y in range(20):
            if grid[x][y] > 0:
                height = 20 - y
                break
        sum_height += height
    return sum_height
def cal_bumpiness(grid):
    height_list = []
    for x in range(10):
        height = 0
        for y in range(20):
            if grid[x][y] > 0:
                height = 20 - y
                break
        height_list.append(height)
    bumpiness = 0
    for x in range(len(height_list)-1):
        bumpiness += abs(height_list[x] - height_list[x+1])
    return bumpiness




if __name__ == '__main__':
    test = 'single'
    T = Tetris(action_type = test,use_fitness = False,is_display = True)
    state = T.reset()
    if test == 'single':
        actionlist = ['Right','Left','Down','Rotate','HardDrop','Hold']
        print(actionlist)
        while True:
            actionID = input('key : ')
            if actionID not in ['0','1','2','3','4','5']:
                continue
            state,reward,done,_,_ = T.step(int(actionID))
            T.test()
            print(T.cal_fitness(T.grid))
            #T.draw()
            print("Reward: " + str(reward))
            if done:
                print('Game Over.')
    else:
        valid = [ str(x) for x in range(53) ]
        while True:
            actionID = input('key : ')
            if actionID not in valid:
                continue
            state,_,done,_ = T.step(int(actionID))
            T.draw()
            if done:
                print('Game Over.')
=======
#!/usr/bin/env python2
#-*- coding: utf-8 -*-

# NOTE FOR WINDOWS USERS:
# You can download a "exefied" version of this game at:
# http://hi-im.laria.me/progs/tetris_py_exefied.zip
# If a DLL is missing or something like this, write an E-Mail (me@laria.me)
# or leave a comment on this gist.

# Very simple tetris implementation
# 
# Control keys:
#       Down - Drop stone faster
# Left/Right - Move stone
#         Up - Rotate Stone clockwise
#     Escape - Quit game
#          P - Pause game
#     Return - Instant drop
#
# Have fun!

# Copyright (c) 2010 "Laria Carolin Chabowski"<me@laria.me>
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from random import randrange as rand
import pygame, sys

# The configuration
cell_size =	18
cols =		10
rows =		22
maxfps = 	30

colors = [
(0,   0,   0  ),
(255, 85,  85),
(100, 200, 115),
(120, 108, 245),
(255, 140, 50 ),
(50,  120, 52 ),
(146, 202, 73 ),
(150, 161, 218 ),
(35,  35,  35) # Helper color for background grid
]

# Define the shapes of the single parts
tetris_shapes = [
	[[1, 1, 1],
	 [0, 1, 0]],
	
	[[0, 2, 2],
	 [2, 2, 0]],
	
	[[3, 3, 0],
	 [0, 3, 3]],
	
	[[4, 0, 0],
	 [4, 4, 4]],
	
	[[0, 0, 5],
	 [5, 5, 5]],
	
	[[6, 6, 6, 6]],
	
	[[7, 7],
	 [7, 7]]
]

def rotate_clockwise(shape):
	return [ [ shape[y][x]
			for y in range(len(shape)) ]
		for x in range(len(shape[0]) - 1, -1, -1) ]

def check_collision(board, shape, offset):
	off_x, off_y = offset
	for cy, row in enumerate(shape):
		for cx, cell in enumerate(row):
			try:
				if cell and board[ cy + off_y ][ cx + off_x ]:
					return True
			except IndexError:
				return True
	return False

def remove_row(board, row):
	del board[row]
	return [[0 for i in range(cols)]] + board
	
def join_matrixes(mat1, mat2, mat2_off):
	off_x, off_y = mat2_off
	for cy, row in enumerate(mat2):
		for cx, val in enumerate(row):
			mat1[cy+off_y-1	][cx+off_x] += val
	return mat1

def new_board():
	board = [ [ 0 for x in range(cols) ]
			for y in range(rows) ]
	board += [[ 1 for x in range(cols)]]
	return board

class TetrisApp(object):
	def __init__(self):
		pygame.init()
		pygame.key.set_repeat(250,25)
		self.width = cell_size*(cols+6)
		self.height = cell_size*rows
		self.rlim = cell_size*cols
		self.bground_grid = [[ 8 if x%2==y%2 else 0 for x in range(cols)] for y in range(rows)]
		
		self.default_font =  pygame.font.Font(
			pygame.font.get_default_font(), 12)
		
		self.screen = pygame.display.set_mode((self.width, self.height))
		pygame.event.set_blocked(pygame.MOUSEMOTION) # We do not need
		                                             # mouse movement
		                                             # events, so we
		                                             # block them.
		self.next_stone = tetris_shapes[rand(len(tetris_shapes))]
		self.init_game()
	
	def new_stone(self):
		self.stone = self.next_stone[:]
		self.next_stone = tetris_shapes[rand(len(tetris_shapes))]
		self.stone_x = int(cols / 2 - len(self.stone[0])/2)
		self.stone_y = 0
		
		if check_collision(self.board,
		                   self.stone,
		                   (self.stone_x, self.stone_y)):
			self.gameover = True
	
	def init_game(self):
		self.board = new_board()
		self.new_stone()
		self.level = 1
		self.score = 0
		self.lines = 0
		pygame.time.set_timer(pygame.USEREVENT+1, 1000)
	
	def disp_msg(self, msg, topleft):
		x,y = topleft
		for line in msg.splitlines():
			self.screen.blit(
				self.default_font.render(
					line,
					False,
					(255,255,255),
					(0,0,0)),
				(x,y))
			y+=14
	
	def center_msg(self, msg):
		for i, line in enumerate(msg.splitlines()):
			msg_image =  self.default_font.render(line, False,
				(255,255,255), (0,0,0))
		
			msgim_center_x, msgim_center_y = msg_image.get_size()
			msgim_center_x //= 2
			msgim_center_y //= 2
		
			self.screen.blit(msg_image, (
			  self.width // 2-msgim_center_x,
			  self.height // 2-msgim_center_y+i*22))
	
	def draw_matrix(self, matrix, offset):
		off_x, off_y  = offset
		for y, row in enumerate(matrix):
			for x, val in enumerate(row):
				if val:
					pygame.draw.rect(
						self.screen,
						colors[val],
						pygame.Rect(
							(off_x+x) *
							  cell_size,
							(off_y+y) *
							  cell_size, 
							cell_size,
							cell_size),0)
	
	def add_cl_lines(self, n):
		linescores = [0, 40, 100, 300, 1200]
		self.lines += n
		self.score += linescores[n] * self.level
		if self.lines >= self.level*6:
			self.level += 1
			newdelay = 1000-50*(self.level-1)
			newdelay = 100 if newdelay < 100 else newdelay
			pygame.time.set_timer(pygame.USEREVENT+1, newdelay)
	
	def move(self, delta_x):
		if not self.gameover and not self.paused:
			new_x = self.stone_x + delta_x
			if new_x < 0:
				new_x = 0
			if new_x > cols - len(self.stone[0]):
				new_x = cols - len(self.stone[0])
			if not check_collision(self.board,
			                       self.stone,
			                       (new_x, self.stone_y)):
				self.stone_x = new_x
	def quit(self):
		self.center_msg("Exiting...")
		pygame.display.update()
		sys.exit()
	
	def drop(self, manual):
		if not self.gameover and not self.paused:
			self.score += 1 if manual else 0
			self.stone_y += 1
			if check_collision(self.board,
			                   self.stone,
			                   (self.stone_x, self.stone_y)):
				self.board = join_matrixes(
				  self.board,
				  self.stone,
				  (self.stone_x, self.stone_y))
				self.new_stone()
				cleared_rows = 0
				while True:
					for i, row in enumerate(self.board[:-1]):
						if 0 not in row:
							self.board = remove_row(
							  self.board, i)
							cleared_rows += 1
							break
					else:
						break
				self.add_cl_lines(cleared_rows)
				return True
		return False
	
	def insta_drop(self):
		if not self.gameover and not self.paused:
			while(not self.drop(True)):
				pass
	
	def rotate_stone(self):
		if not self.gameover and not self.paused:
			new_stone = rotate_clockwise(self.stone)
			if not check_collision(self.board,
			                       new_stone,
			                       (self.stone_x, self.stone_y)):
				self.stone = new_stone
	
	def toggle_pause(self):
		self.paused = not self.paused
	
	def start_game(self):
		if self.gameover:
			self.init_game()
			self.gameover = False
	
	def run(self):
		key_actions = {
			'ESCAPE':	self.quit,
			'LEFT':		lambda:self.move(-1),
			'RIGHT':	lambda:self.move(+1),
			'DOWN':		lambda:self.drop(True),
			'UP':		self.rotate_stone,
			'p':		self.toggle_pause,
			'SPACE':	self.start_game,
			'RETURN':	self.insta_drop
		}
		
		self.gameover = False
		self.paused = False
		
		dont_burn_my_cpu = pygame.time.Clock()
		while 1:
			self.screen.fill((0,0,0))
			if self.gameover:
				self.center_msg("""Game Over!\nYour score: %d
Press space to continue""" % self.score)
			else:
				if self.paused:
					self.center_msg("Paused")
				else:
					pygame.draw.line(self.screen,
						(255,255,255),
						(self.rlim+1, 0),
						(self.rlim+1, self.height-1))
					self.disp_msg("Next:", (
						self.rlim+cell_size,
						2))
					self.disp_msg("Score: %d\n\nLevel: %d\
\nLines: %d" % (self.score, self.level, self.lines),
						(self.rlim+cell_size, cell_size*5))
					self.draw_matrix(self.bground_grid, (0,0))
					self.draw_matrix(self.board, (0,0))
					self.draw_matrix(self.stone,
						(self.stone_x, self.stone_y))
					self.draw_matrix(self.next_stone,
						(cols+1,2))
			pygame.display.update()
			
			for event in pygame.event.get():
				if event.type == pygame.USEREVENT+1:
					self.drop(False)
				elif event.type == pygame.QUIT:
					self.quit()
				elif event.type == pygame.KEYDOWN:
					for key in key_actions:
						if event.key == eval("pygame.K_"
						+key):
							key_actions[key]()
					
			dont_burn_my_cpu.tick(maxfps)

if __name__ == '__main__':
	App = TetrisApp()
	App.run()
>>>>>>> 201ed3e23cd9f1bd06bd11990f9ca51cc403714f
