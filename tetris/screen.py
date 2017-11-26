import pygame
from draw import *
from move import ipieces,opieces,jpieces,lpieces,zpieces,spieces,tpieces,zeropieces,allpieces

maxfps=30

class Screen:
    def __init__(self,save_video = False):
        pygame.init()
        self.screen = pygame.display.set_mode((800,600))#screen is 800*600
        self.screen.blit(gamescreen,(0,0))#blitting the main background
        pygame.display.flip()
    def drawScreen(self,grid,px,py,hdpx,hdpy,block,held,nextlist,positions,sent,step_cnt,is_p2 = False,reset=False):
        bias = 0
        if is_p2:
            bias += 383
        if reset:
            self.screen.blit(gamescreen,(0,0))#blitting the main background
        self.drawHeld(grid,held,46+8+bias,161)#draws held piece for grid1
        self.drawNext(grid,nextlist,318+bias,161)#draws next piece for grid1
        self.drawNumbers(grid,sent,56+bias,377)#draws the linessent on the screen for grid1
        self.drawBackground(112+bias,138,grid,positions) # draw background
        self.drawGhostPiece(112+bias,138,block,hdpx,hdpy)
        self.drawPiece(112+bias,138,block,px,py) #drawing the pieces
        self.drawBoard(grid,112+bias,138) #drawing the grid
        pygame.display.flip()
        #pygame.image.save(self.screen,'images/step' + str(step_cnt))
    def drawHeld(self,grid,held,sx,sy):
        if held != zeropieces:
            num=allpieces.index(held)
            pos=[]
            for x in range (4):
                for y in range (4):
                    if held[0][x][y]>0:
                        pos.append((x,y))
            self.screen.blit(holdback,(sx-8,159))
            if num>1:
                for i in range(4):#if its an i piece different x and y position
                    self.screen.blit(resizepics[num],
                                     (sx+int(pos[i][0]*12),sy+int(pos[i][1]*12)))
            if num==0:
                for i in range(4):#if its an o piece different x and y position
                    self.screen.blit(resizepics[num],
                                     (sx-5+int(pos[i][0]*12),sy-6+int(pos[i][1]*12)))
            if num==1:
                for i in range(4):#any other piece id the same x and y position
                    self.screen.blit(resizepics[num],
                                     (sx-5+int(pos[i][0]*12),sy+int(pos[i][1]*12)))
    def drawNext(self,grid,nextpieces,sx,sy): 
        for i in range (5):#5 different pieces 
            pos=[]
            for x in range (4):
                for y in range (4): #same procedure as the drawhed function
                    if nextpieces[i][0][x][y]>0:
                        num=nextpieces[i][0][x][y]
                        pos.append((x,y))
            
            if i==0: #position 1
                self.screen.blit(holdback,(sx-1,159))            
                if num ==1:#i piece is different x and y pos
                    for i in range (4):
                        self.screen.blit(resizepics[num-1],
                                         (sx+1+int(pos[i][0]*12),156+int(pos[i][1]*12)))
                elif num==2: #o piece is different x and y pos
                    for i in range (4):
                        self.screen.blit(resizepics[num-1],
                                         (sx+1+int(pos[i][0]*12),158+int(pos[i][1]*12)))                
                else: #every other piece is the same x and y pos              
                    for i in range (4):
                        self.screen.blit(resizepics[num-1],
                                         (sx+7+int(pos[i][0]*12),159+int(pos[i][1]*12)))
            if i==1: #position2
                self.screen.blit(nextback2,(sx+2,230))
                if num==1:#i piece is differnet x and y pos
                    for i in range (4):
                        self.screen.blit(nextpics[num-1],
                                         (sx+9+int(pos[i][0]*8),235+int(pos[i][1]*8)))
                if num==2:#o piece is differnt x and y pos
                    for i in range (4):
                        self.screen.blit(nextpics[num-1],
                                         (sx+10+int(pos[i][0]*8),235+int(pos[i][1]*8)))
                if num>2:#every other piece same x and y pos
                    for i in range (4):
                        self.screen.blit(nextpics[num-1],
                                         (sx+13+int(pos[i][0]*8),235+int(pos[i][1]*8)))
            if i>=2:#position 3,4,5
                self.screen.blit(nextback3,(sx+4,288+52*(i-2)))
                if num==1: #same as above
                    for j in range (4):
                        self.screen.blit(nextpics[num-1],
                                         (sx+9+int(pos[j][0]*8),288+(i-2)*51+int(pos[j][1]*8)))
                if num==2: #same as above
                    for j in range (4):
                        self.screen.blit(nextpics[num-1],
                                         (sx+9+int(pos[j][0]*8),292+(i-2)*51+int(pos[j][1]*8)))
                if num>2: #same as above
                    for j in range (4):
                        self.screen.blit(nextpics[num-1],
                                         (sx+12+int(pos[j][0]*8),292+(i-2)*51+int(pos[j][1]*8)))
    def drawBlock(self,sx,sy,x,y,val):
        pics = [ipiece,opiece,jpiece,lpiece,zpiece,spiece,tpiece,lspiece]
        self.screen.blit(pics[val-1],(sx+(x)*18,sy+(y)*18))
    def drawPiece(self,sx,sy,block,px,py):
        for x in range(4):
            for y in range (4):
                if block[x][y]>0:
                    if -1<px+x<10 and -1<py+y<20:
                        self.drawBlock(sx,sy,px+x,py+y,block[x][y])
    def drawGhostPiece(self,sx,sy,block,px,py):
        for x in range(4):
            for y in range (4):
                if block[x][y]>0:
                    if -1<px+x<10 and -1<py+y<20:
                        self.screen.blit(ghost,(sx+(px+x)*18,sy+(py+y)*18))
    def drawBoard(self,grid,sx,sy):
        for x in range (10):
            for y in range (20):
                if grid[x][y]>0:
                    self.drawBlock(sx,sy,x,y,grid[x][y])
    def drawBackground(self,sx,sy,grid,positions):
        for x in range(10):
            for y in range (20):
                if grid[x][y] == 0 and (x,y) not in positions:
                    if (x+y)%2 == 0:
                        self.screen.blit(dgrey,(sx+x*18,sy+y*18))
                    elif (x+y)%2 == 1:
                        self.screen.blit(lgrey,(sx+x*18,sy+y*18))
    def drawNumbers(self,grid,sent,sx,sy):
        tens=sent // 10#integer division tens digit
        ones=sent%10#remainder ones digit
        self.screen.blit(sentback,(sx-12,sy))
        if tens>0:
            if tens==1:
                #blitting the numbers at the poisition in numbers list
                self.screen.blit(numbers[tens],(sx-14,sy))
                self.screen.blit(numbers[ones],(sx+7,sy))
            else:
                self.screen.blit(numbers[tens],(sx-14,sy))
                self.screen.blit(numbers[ones],(sx+14,sy))
        else:
            self.screen.blit(numbers[ones],(sx,sy))
