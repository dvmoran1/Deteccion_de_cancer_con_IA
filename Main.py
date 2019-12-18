import cv2
import numpy as np
import math as mt
from pybrain3.tools.shortcuts import buildNetwork
from pybrain3.datasets import SupervisedDataSet
from pybrain3.supervised.trainers import BackpropTrainer
import statistics as stats


def cal_momentos(img_procesada):
    momentos = []
    Hu = cv2.moments(img_procesada)
    for k in Hu:
        # print(Hu.get(i))
        momentos.append(Hu.get(k))
    #print(momentos)
    # for l in range(0,24):
    #     momentos[l] = -1 * mt.copysign(1.0, momentos[l]) * mt.log10(abs(momentos[l]))
    #     #print(l)
    return momentos

def detect_laterales(contours):
    miny = 9999
    area=0
    index = -1
    for i in range(0, len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        #print(y)
        if ((y < miny and y < 100) and (w*h > area and w*h <150000)):
            miny = y
            index = i
            area =(w*h)
            #print("aqui regresa")
    #print(str(w*h) +"w:"+str(w))
    return index

def procesado(image,x,y,r):
    kernel = np.ones((3, 3), np.uint8)
    # width = int(image.shape[1] * .5)
    # height = int(image.shape[0] * .5)
    # dim = (width, height)
    # P1 = cv2.resize(image, dim)
    # cv2.imshow("imagen1", P1)
    #erosion = cv2.erode(image, kernel, iterations=3)
    r = int(r*1.2)
    img_rec = image[(1024 - y - r):(1024-y+r), (x-r):(x+r)]
    erosion = cv2.blur(img_rec, (3, 3))
    dilatado = cv2.dilate(erosion, kernel, iterations=1)

    ret, otsu = cv2.threshold(erosion, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret, dila = cv2.threshold(dilatado, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(dila, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #
    areas = [cv2.contourArea(ctr) for ctr in contours]
    max_contour = [contours[areas.index(max(areas))]]
    area= max(areas)
    #print(area)
    perimetro = cv2.arcLength(contours[areas.index(max(areas))], True)
    #print(perimetro)
    circularidad = (4*mt.pi)*(round(area,2)/perimetro*perimetro)
    #print('circularidad: ',circularidad)
    his= cv2.calcHist([dila],[0],None,[255],[1,256])
    media = 0
    for i in range(0,len(his)):
        media+=his[i]
    media = media/len(his)
    des_stand = 0
    for i in range(0,len(his)):
        des_stand = des_stand + ((his[i]-media)**2)
    des_stand = mt.sqrt(des_stand/len(his))

    # cv2.imshow("imagen", erosion)
    # cv2.imshow("imagen libre", dilatado)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    momentos=[]
    momentos = cal_momentos(dila)
    momentos.append(area)
    momentos.append(perimetro)
    momentos.append(circularidad)
    momentos.append(media)
    momentos.append(des_stand)
    #print(momentos)

    return momentos


net = buildNetwork(29, 5, 1, bias=True)
ds = SupervisedDataSet(29, 1)
veredicto=""

output = [[0],[1]]
Ninput = [21,21]
puntos=[]
benigno =[[535,425,197],[522,280,69],[477,133,30],[525,425,33],[471,458,40],[667,365,31],[595,864,68],[547,573,48],[653,477,49],[493,125,49],[674,443,79],[322,676,43],[388,742,66],[546,463,33],[462,406,44],[432,149,20],[492,473,131],[544,194,38],[680,494,20],[612,297,34],[714,340,23],[357,365,50],[600,621,111],[492,434,87],[191,549,23],[523,551,48],[252,788,52],[347,636,26],[669,543,49],[351,661,62]]
maligno = [[538,681,29],[338,314,56],[318,359,27],[266,517,28],[468,717,23],[510,547,49],[423,662,43],[415,460,38],[516,279,98],[190,427,51],[505,575,107],[461,532,117],[480,576,84],[423,262,79],[366,620,33],[700,552,60],[220,552,28],[469,728,49],[470,759,29],[313,540,27],[326,607,174],[540,565,88],[489,480,82],[462,627,62],[492,600,70],[600,514,67],[519,362,54],[352,624,114],[403,524,47],[557,772,37]]
puntos=benigno
for i in range(0,2):
    if i==1:
        puntos = maligno
    for j in range(1,Ninput[i]):
        #print(str(i)+':'+str(j))
        #image = cv2.imread("all-mias/mdb076.pgm")
        image = cv2.imread('all-mias/'+str(i)+'/('+str(j)+').pgm',0)
        if image is None:
            print("no se encontro")
        else:
            momentos = procesado(image,puntos[j-1][0],puntos[j-1][1],puntos[j-1][2])
            #momentos = cal_momentos(img_procesada)
            ds.addSample(momentos, output[i])

trainer = BackpropTrainer(net, ds)

er = round(trainer.train(), 3)
#print(er)
while er <= 0.113:
    er = round(trainer.train(), 3)
    print(er)

puntos=benigno
for i in range(17,27):
    img_test = cv2.imread('C:/Users/David VM/Downloads/all-mias/'+str(0)+'/('+str(i)+').pgm',0)
    carat = procesado(img_test,puntos[i-1][0],puntos[i-1][1],puntos[i-1][2])
    #carat = cal_momentos(img_p)
    compute = net.activate(carat)
    result = round(compute[0], 0)
    #print(result)

    if result <= 0:
        veredicto = "Benigno"
    else:
        veredicto = "Maligno"

    print(str(i)+'  '+str(veredicto))
#
#     bgr = cv2.cvtColor(img_p, cv2.COLOR_GRAY2RGB)
#     width = int(bgr.shape[1] * .5)
#     height = int(bgr.shape[0] * .5)
#     dim = (width, height)
#     P1 = cv2.resize(bgr, dim)
#     cv2.imshow(veredicto, P1)
#     cv2.waitKey()
#     cv2.destroyAllWindows()
#             cv2.imshow("imagen", img_procesada)
#             cv2.waitKey()
#             cv2.destroyAllWindows()