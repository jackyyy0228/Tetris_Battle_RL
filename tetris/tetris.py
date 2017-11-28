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
        self.totalSent = 0
        self.totalCleared = 0
        self.step_cnt = 0
        self.state_list_actions = None
        if self.is_display:
            self.display(reset=True)
        return self.get_state(self.grid)
    def step(self,action):
        if self.check_collide(self.piece):
            self.done = True
        if self.done:
            if self.use_fitness:
                return self.get_state(self.grid),np.full([53],GAME_OVER_REWARD),self.done,0,0
            else:
                return self.get_state(self.grid),0,self.done,0,0
        self.step_cnt += 1
        if self.action_type == 'grouped':
            return self.groupedAction(action)
        elif self.action_type == 'single': #haven't finished
            return self.singleAction(action)
        elif self.action_type == 'oracle':
            return self.oracleAction(action)
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
        line_cleared = 0
        line_sent = 0
        reward = 0
        if self.piece.is_final:
            self.change = False
            line_sent,line_cleared = self.check_end()
            reward = line_cleared
            if not self.done:
                self.newBlock()
        if self.is_display:
            self.display()
        self.prev_action = actionID
        return self.get_state(self.grid),reward,self.done,line_sent,line_cleared
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
    def check_tspin(self,piece,is_oracle = False):
        piece_type = piece.piece_type()
        piece_index = piece._index()
        tspin_type = None
        if is_oracle:
            self.prev_action = 3
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
    def check_end(self,is_oracle = False,oracle_piece = None): # return the line sent
        ##check tspin
        if not is_oracle:
            self.grid = self.place_piece(self.grid,self.piece)
            tspin = self.check_tspin(self.piece)
            grid = self.grid
            piece = self.piece
        else:
            grid = self.place_piece(self.grid,oracle_piece)
            tspin = self.check_tspin(oracle_piece)
            piece = oracle_piece
        ## check line cleared
        total_clear = 0
        grid,total_clear = cal_lines(grid)
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
        if not is_oracle:
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
                if piece.block[x][y] > 0:
                    if piece.py+y < 0:
                        if not is_oracle:
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
        sorted(positions, key=lambda pos: pos[1]*200+pos[0])
        return positions
    def place_piece(self,grid,piece):
        temp_grid = copy.deepcopy(grid)
        for x in range(4):
            for y in range(4):
                if piece.block[x][y] > 0:
                    if 10 > piece.px + x >= 0 and 20 > piece.py + y >=0:
                        temp_grid[piece.px + x][piece.py+y] = piece.block[x][y]
        return temp_grid
    def get_state(self,grid,modify_last_element = False): #[grid,nextlist[0]....,held]
        statelist=[]
        #temp_grid = self.place_piece(self.grid,self.piece)
        height = cal_height(grid)
        bumpiness = cal_bumpiness(grid)
        holes = cal_holes(grid)
        temp_grid = np.array(grid)
        grid = np.where( temp_grid > 0 ,1,0)
        grid = np.reshape(grid.T,(20,10,1))
        templist = copy.deepcopy(self.nextlist)
        templist.insert(0,self.piece.b)
        for idx,piece in enumerate(templist):
            if idx == 5 and modify_last_element:
                X = np.full([7],1/7.0)
            else:
                num = allpieces.index(piece)
                X = np.zeros(len(allpieces))
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
    def get_all_possible_states(self): #bfs all possible move
        record = self.bfs(self.piece,False)
        flag = False
        if self.held != zeropieces :
            flag = True
            b = self.held
            piece = Piece(b,b[0],4,-2)
            record_held = self.bfs(piece,True)
        else:
            b = self.nextlist[0]
            piece = Piece(b,b[0],4,-2)
            record_held = self.bfs(piece,True)
        state1_list = []
        state2_list = []
        reward_list = []
        self.state_list_actions = []
        next_list,piece_list,piece_backup,held_backup = copy.deepcopy([self.nextlist,self.piecelist,self.piece,self.held]) #backup
        self.newBlock()
        for piece,actions in record:
            temp_grid = self.place_piece(self.grid,piece)
            state1,state2 = self.get_state(temp_grid,True)
            state1_list.append(state1)
            state2_list.append(state2)
            reward_list.append(self.check_end(True,piece)[1])
            self.state_list_actions.append(actions)
        self.hold()
        for piece,actions in record_held:
            temp_grid = self.place_piece(self.grid,piece)
            if flag :
                state1,state2 = self.get_state(temp_grid,False)
                state1_list.append(state1)
                state2_list.append(state2)
            else:
                state1,state2 = self.get_state(temp_grid,True)
                state1_list.append(state1)
                state2_list.append(state2)
            reward_list.append(self.check_end(True,piece)[1])
            self.state_list_actions.append(actions)
        self.nextlist,self.piecelist,self.piece,self.held = next_list,piece_list,piece_backup,held_backup #backup
        return state1_list,state2_list,reward_list
    def test(self):
        state_list,state2_list,reward_list = self.get_all_possible_states()
        print(reward_list)
        for grid,Y,reward in zip(state_list,state2_list,reward_list):
            for i in range(20):
                for j in range(10):
                    self.grid[j][i] = grid[i][j][0]
                    print(grid[i][j][0],end='')
                print()
            print(Y)
            print("Rewards:",reward)
            self.display()
            import time
            time.sleep(1)
    def oracleAction(self,actionID):
        if not self.state_list_actions:
            print('Please call get_all_possible_state first')
            exit()
        actions = self.state_list_actions[actionID]
        for action in actions:
            state,reward,done,line_sent,line_cleared = self.singleAction(action)
        return state,reward,done,line_sent,line_cleared
    def bfs(self,piece,is_hold):
        start = piece.get_state() #(px,py,rotate_index)
        visited = set()
        record_visited = set() #(positions)
        if is_hold:
            queue = [(start,-1,6)] # (state,prev_idx,action)
        else:
            queue = [(start,-1,-1)] # (state,prev_idx,action)
        record = [] #pieces
        idx = 0
        while idx < len(queue):
            now_state,_,_ = queue[idx]
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
                    positions = self.getPositions(new_piece)
                    positions = tuple(positions)
                    if positions not in record_visited :
                        record_visited.add(positions)
                        visited.add(next_state)
                        actions = [actionID]
                        now_idx = idx -1
                        while True:
                            #back trace
                            _,prev_idx,action = queue[now_idx]
                            actions.append(action)
                            if prev_idx == -1:
                                if action == 6:
                                    actions.append(6)
                                break
                            now_idx = prev_idx
                        actions.reverse() 
                        record.append((new_piece,actions))
                        continue
                queue.append((next_state,idx-1,actionID))
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
    test = 'oracle'
    T = Tetris(action_type = test,use_fitness = False,is_display = True)
    state = T.reset()
    if test == 'single':
        actionlist = ['Right','Left','Down','Rotate','CounterRotate','HardDrop','Hold']
        print(actionlist)
        while True:
            actionID = input('key : ')
            if actionID not in ['0','1','2','3','4','5','6','7']:
                continue
            if actionID == '7':
                T.test()
            else:
                state,reward,done,_,_ = T.step(int(actionID))
            #print(T.cal_fitness(T.grid))
            #print("Reward: " + str(reward))
            if done:
                print('Game Over.')
    elif test == 'grouped':
        valid = [ str(x) for x in range(53) ]
        while True:
            actionID = input('key : ')
            if actionID not in valid:
                continue
            state,_,done,_,_ = T.step(int(actionID))
            T.draw()
            if done:
                print('Game Over.')
    elif test == 'oracle':
        while True:
            states,_,_ = T.get_all_possible_states()
            valid = [ str(x) for x in range(len(states)) ]
            actionID = input('key : ')
            if actionID not in valid:
                continue
            state,_,done,_,_ = T.step(int(actionID))
            T.draw()
            if done:
                print('Game Over.')
