import pygame
from tetris import Tetris

key_actions=['RIGHT','LEFT','DOWN','UP','z','SPACE','LSHIFT']
delay_time= [ 150,150,200,200,200,400,450]
class Keyboard:
    def __init__(self):
        self.T = Tetris(action_type = 'single',is_display = True)
    def run(self):
        while True:
            state = self.T.reset()
            done = False
            while not done:
                flag = False
                for evt in pygame.event.get():
                    if evt.type == pygame.QUIT:
                        exit()
                    if evt.type == pygame.KEYDOWN:
                        for idx,key in enumerate(key_actions):
                            if evt.key == eval("pygame.K_"+key):
                                state,_,done,_,_ = self.T.step(idx)
                                flag = True
                                pygame.time.wait(delay_time[idx])
                keys_pressed = pygame.key.get_pressed()
                if not flag:
                    for idx,key in enumerate(key_actions):
                        key = eval("pygame.K_"+key)
                        if keys_pressed[key]:
                            state,_,done,_,_ = self.T.step(idx)
                            if idx == [3,4,5,6]:
                                pygame.time.wait(delay_time[idx])
                pygame.time.wait(10)

if __name__ == "__main__":
    K = Keyboard()
    K.run()
                


           
                
