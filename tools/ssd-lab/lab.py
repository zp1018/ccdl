#coding=utf-8
import os
import cv2
import sys
import math
import random

##############################################
# ssd lab v1.3
# wish
# 2017年8月12日 16:59:37

#parameters
path = "mo-xiao-fan-lan"
wndName = "ESC close, s save, c clean, p back"
className = ["028", "004"]
colors = [(0, 255, 0), (255, 0, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255)]

#code
currentClass = 0
imgs = os.listdir(path)
imgs = [(imgs[i], path + "/" + imgs[i]) for i in range(len(imgs))]
for i in range(len(imgs)-1, -1, -1):
    if not imgs[i][1].lower().endswith("jpg") and not imgs[i][1].lower().endswith("png") and not imgs[i][1].lower().endswith("jpeg"):
        del imgs[i]


if len(sys.argv) > 1 and sys.argv[1] == "r":
    random.shuffle(imgs)

show = []
cach = []
cach2 = []
dpoint = []
epoint = []
objs = []
waitSecDown = False

def refreshCurrentShow():
    cv2.resize(show, (cach.shape[1], cach.shape[0]), cach)
    cv2.putText(cach, className[currentClass], (10, 30), 1, 2, colors[currentClass], 2)
    drawObjs(objs, cach)

    if waitSecDown:
        cv2.circle(cach, dpoint, 5, colors[currentClass], 2)
    cv2.imshow(wndName, cach)

def l2dis(a, b):
    return math.sqrt(a*a+b*b)

def onMouse(event, x, y, flag, points):
    global dpoint, waitSecDown, wndName, epoint, cach

    if event == cv2.EVENT_LBUTTONDOWN:
        if waitSecDown:
            epoint = (x, y)
            waitSecDown = False
            objs.append((dpoint, epoint, currentClass))
        else:
            dpoint = (x, y)
            waitSecDown = True

        refreshCurrentShow()
    elif event == cv2.EVENT_MOUSEMOVE:
        cv2.resize(cach, (cach.shape[1], cach.shape[0]), cach2)
        cv2.line(cach2, (0, y), (cach.shape[1], y), (0, 255, 0), 2)
        cv2.line(cach2, (x, 0), (x, cach.shape[0]), (0, 255, 0), 2)
        cv2.imshow(wndName, cach2)

    elif event == cv2.EVENT_RBUTTONDOWN:
        vd = []
        invd = 9999999
        for i in range(len(objs)):
            cx = (objs[i][0][0] + objs[i][1][0]) * 0.5
            cy = (objs[i][0][1] + objs[i][1][1]) * 0.5
            xmin = objs[i][0][0] if objs[i][0][0] < objs[i][1][0] else objs[i][1][0]
            xmax = objs[i][0][0] if objs[i][0][0] > objs[i][1][0] else objs[i][1][0]
            ymin = objs[i][0][1] if objs[i][0][1] < objs[i][1][1] else objs[i][1][1]
            ymax = objs[i][0][1] if objs[i][0][1] > objs[i][1][1] else objs[i][1][1]
            if x >= xmin and x <= xmax and y >= ymin and y <= ymax:
                vd.append(l2dis(cx - x, cy - y))
                #del objs[n]
            else:
                vd.append(invd)

        if len(vd) > 0:
            n = vd.index(min(vd))
            if(vd[n] != invd):
                del objs[n]
                refreshCurrentShow()

def drawObjs(objs, canvas):
    for item in objs:
        d = item[0]
        e = item[1]
        c = colors[item[2]]
        cv2.rectangle(canvas, d, e, c, 5)

def loadObjs(name):
    if not os.path.exists(name):
        return 0, []

    obs = []
    cls = 0
    with open(name, "r") as txt:
        info = txt.readline().split(",")
        num = int(info[0])
        cls = int(info[1])
        for i in range(num):
            info = txt.readline().split(",")
            d = (int(info[0]), int(info[1]))
            e = (int(info[2]), int(info[3]))
            c = int(className.index(info[5].split("\n")[0]))
            obs.append((d, e, c))
    return cls, obs

def saveBreakpoint(bkp):
    with open("breakpoint.txt", "w") as p:
        p.write("%d" % bkp)

def loadBreakpoint():
    p = 0

    if os.path.exists("breakpoint.txt"):
        with open("breakpoint.txt", "r") as p:
            p = int(p.read())
    return p

def saveXML(name, objs, cls, w, h):
    with open(name, "w") as xml:
        xml.write("<annotation><size><width>%d</width><height>%d</height></size>" % (w, h))
        for item in objs:
            fmt = """
            <object>
                <name>%s</name>
                <bndbox>
                    <xmin>%d</xmin>
                    <ymin>%d</ymin>
                    <xmax>%d</xmax>
                    <ymax>%d</ymax>
                </bndbox>
            </object>
            """

            d = item[0]
            e = item[1]
            pmin = (min(d[0], e[0]), min(d[1], e[1]))
            pmax = (max(d[0], e[0]), max(d[1], e[1]))

            cls = item[2]
            xml.write(fmt % (className[cls], pmin[0], pmin[1], pmax[0], pmax[1]))

        xml.write("</annotation>")

    with open(name + ".txt", "w") as txt:
        txt.write("%d,%d\n" % (len(objs), cls))
        for item in objs:
            d = item[0]
            e = item[1]
            pmin = (min(d[0], e[0]), min(d[1], e[1]))
            pmax = (max(d[0], e[0]), max(d[1], e[1]))

            cls = item[2]
            txt.write("%d,%d,%d,%d,%d,%s\n" % (pmin[0], pmin[1], pmax[0], pmax[1], cls, className[cls]))



if len(imgs) == 0:
    print "empty imgs dir."
    exit(1)


endOf = False;
i = loadBreakpoint()
while True:
    if i < 0:
        i = len(imgs) + i
    elif i > len(imgs) - 1:
        i = i - len(imgs)

    saveBreakpoint(i)
    show = cv2.imread(imgs[i][1])
    cach = cv2.imread(imgs[i][1])
    cach2 = cv2.imread(imgs[i][1])
    pos = imgs[i][0].rfind(".")
    currentClass, objs = loadObjs("%s/%s.xml.txt" %(path, imgs[i][0][:pos]))
    waitSecDown = False

    while True:
        cv2.resize(show, (cach.shape[1], cach.shape[0]), cach)

        refreshCurrentShow()
        cv2.setMouseCallback(wndName, onMouse)
        key = cv2.waitKey()
        key = key & 0xFF;

        if key == 0x1B: #ESC
            endOf = True;
            break

        if key >= ord('1') and key <= ord('9'):
            currentClass = key - ord('1')
            currentClass = max(currentClass, 0)
            currentClass = min(len(className) - 1, currentClass)
            continue

        if key == ord('c'):
            objs = []
            currentClass = 0
            continue

        if key == 82 or key == 44:   #up
            i = i - 2
            break

        if key == 84 or key == 46:   #down
            break

        if key == ord('p'):
            if len(objs)>0:
                del objs[len(objs)-1]
            continue

        if key == ord('s'): #s
            saveXML("%s/%s.xml" %(path, imgs[i][0][:pos]), objs, currentClass, show.shape[1], show.shape[0])
            break

    if endOf: break;
    i = i+1
