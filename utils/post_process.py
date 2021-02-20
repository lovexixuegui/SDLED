import math
import numpy as np

def findSinglePoints(img,symbol=255,ignore=0):

    x_limit = img.shape[0]-ignore
    y_limit = img.shape[1]-ignore
    singlePoint = []
    tempPoints=[]

    for x in range(x_limit):
        for y in range(y_limit):
            if img[x][y] == symbol:
                neighbor = 0
                #左边
                if x - 1 >= 0 and img[x - 1][y] == symbol:
                    neighbor += 1
                    tempPoints.append((x-1,y))
                # 右边
                if x + 1 < x_limit and img[x + 1][y] == symbol:
                    neighbor += 1
                    tempPoints.append((x+1,y))
                # 上面
                if y - 1 >= 0 and img[x][y - 1] == symbol:
                    neighbor += 1
                    tempPoints.append((x,y-1))
                # 下面
                if y + 1 < y_limit and img[x][y + 1] == symbol:
                    neighbor += 1
                    tempPoints.append((x,y+1))
                # 左上
                if x - 1 >= 0 and y - 1 >= 0 and img[x - 1][y - 1] == symbol:
                    neighbor += 1
                    tempPoints.append((x-1,y-1))
                # 右上
                if x + 1 < x_limit and y - 1 >= 0 and img[x + 1][y - 1] == symbol:
                    neighbor += 1
                    tempPoints.append((x+1,y-1))
                # 右下
                if x + 1 < x_limit and y + 1 < y_limit and img[x + 1][y + 1] == symbol:
                    neighbor += 1
                    tempPoints.append((x+1,y+1))
                # 左下
                if x - 1 >= 0 and y + 1 < y_limit and img[x - 1][y + 1] == symbol:
                    neighbor += 1
                    tempPoints.append((x-1,y+1))
                if neighbor == 1:
                    singlePoint.append((x, y))
                    tempPoints.pop()
                elif neighbor==2:
                    x1,y1=tempPoints.pop()
                    x2,y2=tempPoints.pop()
                    if((x1==x2 and abs(y1-y2)==1)or(y1==y2 and abs(x1-x2)==1)):
                        singlePoint.append((x,y))
                elif neighbor>2:
                    for i in range(neighbor):
                        tempPoints.pop()
    return singlePoint

def findNeedRepair(image,singlePoints,symbol=255):
    x_limit = image.shape[0]
    y_limit = image.shape[1]
    findPoints=[]
    repeatPoints=[]
    index=-1
    needRepair=[]
    img=np.copy(image)

    while len(singlePoints)>0:
        xx,yy=singlePoints.pop()


        if (xx,yy) not in repeatPoints:
            index+=1
            img[xx][yy] = 0
            needRepair.append((xx, yy, index))
            repeatPoints.append((xx, yy))
            findPoints.append((xx,yy))

        while len(findPoints)>0:
            x,y=findPoints.pop()
            img[x][y]=0

            neighbor = 0
            #左边
            if x - 1 >= 0 and img[x - 1][y] == symbol:
                neighbor += 1
                findPoints.append((x - 1, y))
            # 右边
            if x + 1 < x_limit and img[x + 1][y] == symbol:
                neighbor += 1
                findPoints.append((x + 1, y))
            # 上面
            if y - 1 >= 0 and img[x][y - 1] == symbol:
                neighbor += 1
                findPoints.append((x, y - 1))
            # 下面
            if y + 1 < y_limit and img[x][y + 1] == symbol:
                neighbor += 1
                findPoints.append((x, y + 1))
            # 左上
            if x - 1 >= 0 and y - 1 >= 0 and img[x - 1][y - 1] == symbol:
                neighbor += 1
                findPoints.append((x - 1, y - 1))
            # 右上
            if x + 1 < x_limit and y - 1 >= 0 and img[x + 1][y - 1] == symbol:
                neighbor += 1
                findPoints.append((x + 1, y - 1))
            # 右下
            if x + 1 < x_limit and y + 1 < y_limit and img[x + 1][y + 1] == symbol:
                neighbor += 1
                findPoints.append((x + 1, y + 1))
            # 左下
            if x - 1 >= 0 and y + 1 < y_limit and img[x - 1][y + 1] == symbol:
                neighbor += 1
                findPoints.append((x - 1, y + 1))
            if neighbor==0 and (x,y) in singlePoints:
                needRepair.append((x,y,index))
                repeatPoints.append((x,y))
    # print(needRepair)
    return needRepair,index

def pointsDistance(x1,y1,x2,y2):
    distance=math.sqrt((x1-x2)**2+(y1-y2)**2)
    return distance

def repair_limit_edge2(img,x1,y1,x2,y2,edge_limit=3,symbol=255):
    # print("repair_limit_edge")
    x_limit=img.shape[0]-1
    y_limit=img.shape[1]-1

    if (x1<edge_limit and y1<edge_limit):
        if (x2<edge_limit):
            for m in range(0,x2):
                img[m][y2]=symbol
            for n in range(0,x1):
                img[n][y1]=symbol
            for i in range(y1,y2+1):
                img[0][i]=symbol
        elif(y2<edge_limit):
            for m in range(0,y2):
                img[x2][m]=symbol
            for n in range(0,y1):
                img[x1][n]=symbol
            for i in range(x1,x2+1):
                img[i][0]=symbol
    elif(x1>x_limit-edge_limit and y1>y_limit-edge_limit):
        if(x2>x_limit-edge_limit):
            for m in range(x2,x_limit+1):
                img[m][y2] = symbol
            for n in range(x1,x_limit+1):
                img[n][y1] = symbol
            for i in range(y2, y1+1):
                img[x_limit][i] = symbol
        elif (y2 >y_limit-edge_limit):
            for m in range(y2, y_limit+1):
                img[x2][m] = symbol
            for n in range(y1, y_limit+1):
                img[x1][n] = symbol
            for i in range(x2, x1+1):
                img[i][0] = symbol

    elif(x2<edge_limit and y2<edge_limit):
        if (x1<edge_limit):
            for m in range(0,x1):
                img[m][y1]=symbol
            for n in range(0,x2):
                img[n][y2]=symbol
            for i in range(y2,y1+1):
                img[0][i]=symbol
        elif(y1<edge_limit):
            for m in range(0,y1):
                img[x1][m]=symbol
            for n in range(0,y2):
                img[x2][n]=symbol
            for i in range(x2,x1+1):
                img[i][0]=symbol
    elif(x2>x_limit-edge_limit and y2>y_limit-edge_limit):
        if(x1>x_limit-edge_limit):
            for m in range(x1,x_limit+1):
                img[m][y1] = symbol
            for n in range(x2,x_limit+1):
                img[n][y2] = symbol
            for i in range(y1, y2+1):
                img[x_limit][i] = symbol
        elif (y1 >y_limit-edge_limit):
            for m in range(y1, y_limit+1):
                img[x1][m] = symbol
            for n in range(y2, y_limit+1):
                img[x2][n] = symbol
            for i in range(x1, x2+1):
                img[i][0] = symbol


    elif(x1>x_limit-edge_limit and y1<edge_limit):
        if(x2>x_limit-edge_limit):
            for m in range(x2,x_limit+1):
                img[m][y2]=symbol
            for n in range(x1,x_limit):
                img[n][y1]=symbol
            for i in range(y1,y2+1):
                img[x_limit][i]=symbol
        elif(y2<edge_limit):
            for m in range(0,y2):
                img[x2][m]=symbol
            for n in range(0,y1):
                img[x1][n]=symbol
            for i in range(x2,x1):
                img[i][0]=symbol
    elif(x1<edge_limit and y1>y_limit-edge_limit):
        if(x2<edge_limit):
            for m in range(0,x2):
                img[m][y2]=symbol
            for n in range(0,x1):
                img[n][y1]=symbol
            for i in range(y2,y1+1):
                img[0][i]=symbol
        elif(y2>y_limit-edge_limit):
            for m in range(y1,y_limit+1):
                img[x1][m]=symbol
            for n in range(y2,y_limit+1):
                img[x2][n]=symbol
            for i in range(x1,x2+1):
                img[i][y_limit]=symbol

    elif(x2>x_limit-edge_limit and y2<edge_limit):
        if(x1>x_limit-edge_limit):
            for m in range(x1,x_limit+1):
                img[m][y1]=symbol
            for n in range(x2,x_limit):
                img[n][y2]=symbol
            for i in range(y2,y1+1):
                img[x_limit][i]=symbol
        elif(y1<edge_limit):
            for m in range(0,y1):
                img[x1][m]=symbol
            for n in range(0,y2):
                img[x2][n]=symbol
            for i in range(x1,x2):
                img[i][0]=symbol
    elif(x2<edge_limit and y2>y_limit-edge_limit):
        if(x1<edge_limit):
            for m in range(0,x1):
                img[m][y1]=symbol
            for n in range(0,x2):
                img[n][y2]=symbol
            for i in range(y1,y2+1):
                img[0][i]=symbol
        elif(y1>y_limit-edge_limit):
            for m in range(y2,y_limit+1):
                img[x1][m]=symbol
            for n in range(y1,y_limit+1):
                img[x1][n]=symbol
            for i in range(x2,x1+1):
                img[i][y_limit]=symbol

    else:
        if(x1<edge_limit and x2<edge_limit):
            (y_max,y_min)=(y1,y2) if y1>y2 else (y2,y1)
            for m in range(0,x1):
                img[m][y1]=symbol
            for n in range(0,x2):
                img[n][y2]=symbol
            for i in range(y_min,y_max+1):
                img[0][i]=symbol
        elif(y1>y_limit-edge_limit and y2>y_limit-edge_limit):
            (x_max,x_min)=(x1,x2)if x1>x2 else (x2,x1)
            for m in range(y1,y_limit+1):
                img[x1,m]=symbol
            for n in range(y2,y_limit+1):
                img[x2,n]=symbol
            for i in range(x_min,x_max+1):
                img[i][y_limit]=symbol
        elif(x1>x_limit-edge_limit and x2>x_limit-edge_limit):
            (y_max, y_min) = (y1, y2) if y1 > y2 else (y2, y1)
            for m in range(x1,x_limit+1):
                img[m][y1] = symbol
            for n in range(x2,x_limit+1):
                img[n][y2] = symbol
            for i in range(y_min, y_max + 1):
                img[x_limit][i] = symbol
        elif (y1<edge_limit and y2<edge_limit):
            (x_max, x_min) = (x1, x2) if x1 > x2 else (x2, x1)
            for m in range(0,y1):
                img[x1, m] = symbol
            for n in range(0,y2):
                img[x2, n] = symbol
            for i in range(x_min, x_max + 1):
                img[i][0] = symbol

        elif(x1<edge_limit and y2<edge_limit):
            for m in range(0,x1):
                img[m][y1]=symbol
            for n in range(0,y2):
                img[x2][n]=symbol
            for q in range(0,y1):
                img[0][q]=symbol
            for w in range(0,x2):
                img[w][0]=symbol
        elif(y1>y_limit-edge_limit and x2<edge_limit):
            for m in range(y1,y_limit+1):
                img[x1][m]=symbol
            for n in range(0,x2):
                img[n][y2]=symbol
            for q in range(0,x1):
                img[q][y_limit]=symbol
            for w in range(y2,y_limit+1):
                img[0][w]=symbol
        elif (x1 >x_limit- edge_limit and y2 >y_limit- edge_limit):
            for m in range( x1,x_limit+1):
                img[m][y1] = symbol
            for n in range( y2,y_limit+1):
                img[x2][n] = symbol
            for q in range( y1,y_limit+1):
                img[x_limit][q] = symbol
            for w in range(x2,x_limit+1):
                img[w][y_limit] = symbol
        elif (y1 <edge_limit and x2 > x_limit-edge_limit):
            for m in range(0, y1):
                img[x1][m] = symbol
            for n in range(x2, x_limit+1):
                img[n][y2] = symbol
            for q in range(x1, x_limit):
                img[q][0] = symbol
            for w in range(0, y2):
                img[x_limit][w] = symbol

        elif(x2<edge_limit and y1<edge_limit):
            for m in range(0,x2):
                img[m][y2]=symbol
            for n in range(0,y1):
                img[x1][n]=symbol
            for q in range(0,y2):
                img[0][q]=symbol
            for w in range(0,x1):
                img[w][0]=symbol
        elif(y2>y_limit-edge_limit and x1<edge_limit):
            for m in range(y2,y_limit+1):
                img[x2][m]=symbol
            for n in range(0,x1):
                img[n][y1]=symbol
            for q in range(0,x2):
                img[q][y_limit]=symbol
            for w in range(y1,y_limit+1):
                img[0][w]=symbol
        elif (x2 >x_limit- edge_limit and y1 >y_limit- edge_limit):
            for m in range( x2,x_limit+1):
                img[m][y2] = symbol
            for n in range( y1,y_limit+1):
                img[x1][n] = symbol
            for q in range( y2,y_limit+1):
                img[x_limit][q] = symbol
            for w in range(x1,x_limit+1):
                img[w][y_limit] = symbol
        elif (y2 <edge_limit and x1 > x_limit-edge_limit):
            for m in range(0, y2):
                img[x2][m] = symbol
            for n in range(x1, x_limit+1):
                img[n][y1] = symbol
            for q in range(x2, x_limit):
                img[q][0] = symbol
            for w in range(0, y1):
                img[x_limit][w] = symbol

    return img