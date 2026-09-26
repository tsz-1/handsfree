import cv2
import numpy as np
import pyautogui
from cvzone.HandTrackingModule import HandDetector
import time

# ================= 参数配置区 =================
wCam, hCam = 640, 480       
frameR = 150                
smoothening = 8             

pyautogui.PAUSE = 0      
pyautogui.FAILSAFE = False   
wScr, hScr = pyautogui.size() 

# ================= 初始化模块 =================
cap = cv2.VideoCapture(1)
cap.set(3, wCam)
cap.set(4, hCam)

detector = HandDetector(maxHands=1, detectionCon=0.8)

plocX, plocY = 0, 0
clocX, clocY = 0, 0

print("正在启动摄像头，请等待...")

while True:
    success, img = cap.read()
    
    # 【新增防崩溃机制】如果读取失败，打印提示并跳过这一帧，而不是直接崩溃
    if not success or img is None:
        print("警告：无法获取摄像头画面，请检查 Mac 摄像头权限或是否被占用。")
        time.sleep(1) # 暂停1秒防止终端被报错刷屏
        continue      # 跳过后续处理，重新尝试读取下一帧

    # 翻转画面
    img = cv2.flip(img, 1)

    # 寻找手部并获取关键点
    hands, img = detector.findHands(img, flipType=False) 

    if hands:
        hand = hands[0]
        lmList = hand["lmList"] 
        
        if lmList:
            x1, y1 = lmList[8][0], lmList[8][1]
            x2, y2 = lmList[4][0], lmList[4][1]

            fingers = detector.fingersUp(hand)

            cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)

            length, info, img = detector.findDistance((x1, y1), (x2, y2), img)
            cv2.putText(img, f'Pinch Dist: {int(length)}', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

            click_threshold = 50

            if length < click_threshold:
                cv2.circle(img, (info[4], info[5]), 15, (0, 255, 0), cv2.FILLED)
                pyautogui.click()
                time.sleep(0.2)

            elif fingers[1] == 1 and length >= click_threshold:
                x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
                y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
                clocX = plocX + (x3 - plocX) / smoothening
                clocY = plocY + (y3 - plocY) / smoothening
                pyautogui.moveTo(clocX, clocY)
                plocX, plocY = clocX, clocY

    cv2.imshow("Computer Vision Gesture Engine", img)
    
    # 强制将窗口置顶 (Mac 环境下 OpenCV 窗口有时会跑到后台)
    cv2.setWindowProperty("Computer Vision Gesture Engine", cv2.WND_PROP_TOPMOST, 1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()