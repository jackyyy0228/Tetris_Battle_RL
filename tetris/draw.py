from pygame import *
#battlemusic=mixer.Sound("tetris sounds/battlemusic.wav")#importing sound file
#the code below imports many of the images that will be used
gamescreen=image.load("tetris/graphs/background/gamescreen.png")#backscreen
lgrey=image.load("tetris/graphs/background/lightgreysquare.png")#square for grid background
dgrey=image.load("tetris/graphs/background/darkgreysquare.png")#smae as above
ipiece=image.load("tetris/graphs/tetris blocks/lightblue block.png") 
opiece=image.load("tetris/graphs/tetris blocks/yellow block.png")
jpiece=image.load("tetris/graphs/tetris blocks/blue block.png")
lpiece=image.load("tetris/graphs/tetris blocks/orange block.png")
zpiece=image.load("tetris/graphs/tetris blocks/red block.png")
spiece=image.load("tetris/graphs/tetris blocks/green block.png")
tpiece=image.load("tetris/graphs/tetris blocks/purple block.png")
lspiece=image.load("tetris/graphs/tetris blocks/linessent block.png")
sentpiece=image.load("tetris/graphs/tetris blocks/linessent block.png")#dark block for garbage lines
ghost=image.load("tetris/graphs/tetris blocks/ghost block.png") #ghost block 
decimal=image.load("tetris/graphs/tetris numbers/decimal.png") #for timer
ko=image.load("tetris/graphs/tetris icons/KO.png")#knockout image
holdback=image.load("tetris/graphs/tetris icons/holdback.png")#background for pic blitting
sentback=image.load("tetris/graphs/tetris icons/sentback.png")#same as above
nextback2=image.load("tetris/graphs/tetris icons/holdback2.png")#same as above
nextback3=image.load("tetris/graphs/tetris icons/holdback3.png")#same as above
timeback=image.load("tetris/graphs/tetris icons/timeback.png")#same as above
kos = [] #ko pictures
for i in range(1,4):#putting kO pictures in the list 
    kos.append(image.load("tetris/graphs/tetris icons/ko"+str(i)+".png"))

#piecepics list is the list different block pictures
piecepics=[ipiece,opiece,jpiece,lpiece,zpiece,spiece,tpiece,lspiece]
resizepics=[]#blocks will be resized for hold piece 
for i in range (7):
    resizepics.append(transform.smoothscale(piecepics[i],(12,12))) #12 x12 blocks

nextpics=[]#blocks will be resized for next pieces
for i in range (7):
    nextpics.append(transform.smoothscale(piecepics[i],(8,8)))  #8 x8 blocks

numbers=[]#imputing the numbers from 1 to 10 into list 
for i in range (10): #to be used for timer and sent lines
    numbers.append(image.load("tetris/graphs/tetris numbers/"+str(i)+".png"))

combos=[]#inputs the combo pictures
for i in range (1,11):
    combos.append(image.load("tetris/graphs/combo/"+str(i)+"combo.png"))

back=image.load("tetris/graphs/tetris icons/back.png")#the main background screen
tetris=image.load("tetris/graphs/tetris icons/tetris.png")#tetris image

