"""
專題名稱:基於YOLOv8與Homography之羽球員場上定位系統
作者:林冠言

專題描述:
本專題旨在建立一套自動定位羽球選手位置，並且利用選手位置做數據統計
系統流程分為兩大核心：
1.  場地偵測：從影片第一幀中，透過一系列影像處理技術（白點提取、霍夫變換、
    線段精煉、角點合成），計算出能將影像像素對應至標準球場模型的
    單應性變換矩陣 (Homography Matrix)。
2.  選手分析：在後續的影片幀中，使用訓練過的 YOLOv8 模型偵測選手位置，
    並利用前一步計算出的Homography Matrix，將選手的像素座標轉換為
    標準球場上的幾何座標，最後根據其位置進行九宮格區域判斷與統計。
"""

import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt


def white_pixel_extraction(frame, sigma_l, sigma_d, tau):
    """
    從影像中提取可能是球場線的白色像素點(二值化)

    此函式實作l(x,y)進行分類黑、白色像素點，並結合 Harris角點偵測來強化
    線條交接處的特徵，最後建立黑色遮罩僅留下場地的部分

    Args:
        frame(np.array):輸入的BGR格式影像
        sigma_l(int):亮度基準閾值
        sigma_d(int):像素與鄰近點的亮度差異閾值
        tau(int):檢查鄰近點像素的距離

    Returns:
        np.array:一個二值化的遮罩，白點為255，其餘為0
    """

    #步驟一:實作l(x,y)進行分類黑、白色像素點
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    high, width = gray_frame.shape

    white_mask = np.zeros_like(gray_frame)

    for y in range(tau, high - tau):
        for x in range(tau, width - tau):
            pixel = int(gray_frame[y][x])

            if pixel >= sigma_l:
                top_pixel = int(gray_frame[y - tau][x])
                bottom_pixel = int(gray_frame[y + tau][x])
                left_pixel = int(gray_frame[y][x - tau])
                right_pixel = int(gray_frame[y][x + tau])

                if ((pixel - left_pixel) > sigma_d) and ((pixel - right_pixel) > sigma_d):
                    white_mask[y][x] = 255
                elif ((pixel - top_pixel) > sigma_d) and ((pixel - bottom_pixel) > sigma_d):
                    white_mask[y][x] = 255

    #步驟2:Harris角點偵測以強化交接處特徵
    gray_float = np.float32(gray_frame)
    harris_response = cv2.cornerHarris(gray_float, 2, 3, 0.04)
    corner_mask = np.zeros_like(gray_frame)
    corner_mask[harris_response > 0.01 * harris_response.max()] = 255

    combined_mask = cv2.bitwise_or(white_mask, corner_mask)

    #步驟3:建立黑色遮罩
    final_mask = combined_mask.copy()
    final_mask[:, 0:int(width * 0.17)] = 0
    final_mask[:, int(width * 0.83):] = 0
    final_mask[0:int(high * 0.444), :] = 0
    final_mask[int(high * 0.93):, :] = 0

    return final_mask

def PHT(filter_frame, origin_frame, T, minLength, LineGap):
    """
    使用概率霍夫變換(Probabilistic Hough Transform)從二值化遮罩中偵測線段

    Args:
        filter_frame(np.array):輸入的二值化遮罩
        origin_frame(np.array):用於繪製結果的原始影像
        T(int):霍夫變換的閾值
        minLength(int):偵測線段的最小長度
        LineGap(int):線段上兩點之間可容忍的最大間隙

    Returns:
        tuple:繪製了線段的影像,偵測到的線段(np.ndarray)
    """
    lines = cv2.HoughLinesP(filter_frame ,rho = 1.0, theta = np.pi / 180, threshold = T, minLineLength = minLength, maxLineGap = LineGap)
    line_frame = origin_frame.copy()

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
    
    return line_frame, lines

def refine(lines, white_frame, sigma_r):
    """
    透過最小二乘法，對PHT偵測出的線段進行精煉

    Args:
        lines(np.array):PHT輸出的線段
        white_frame(np.array):白色像素遮罩
        sigma_r(int):尋找鄰近點的搜索半徑

    Returns:
        np.array:精煉後的線段
    """
    refine_line = []
    white_pixels = np.argwhere(white_frame > 0)[:,::-1] #先找到白點座標(x,y)

    for line in lines:
        x1, y1, x2, y2 = line[0]

        #步驟1:將線段轉換為法線式 (normal form)
        vx, vy = x2 - x1, y2 - y1 
        nx, ny = -vy, vx 

        n_len = np.sqrt(nx**2 + ny**2) #法向量長度
        if n_len == 0: continue
        nx, ny = nx / n_len, ny / n_len
        d = (nx * x1 + ny * y1)

        #步驟2:找出鄰近的白點集合L
        distance = np.abs(white_pixels @ np.array([nx, ny]) - d)
        L_point = white_pixels[distance < sigma_r]
        
        #步驟3:建立方程組Ax=b並用最小二乘法求解
        A = L_point
        b = np.ones(len(L_point))
        
        try:
            m, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            continue
        mx, my = m[0], m[1]

        #步驟4:將上一步解mx,my轉換回標準線參數
        m_len = np.sqrt(mx**2 + my**2)
        new_d = 1 / m_len
        new_nx, new_ny = mx * new_d, my * new_d
        
        #步驟5:將原始端點投影到新的擬合線上，以保持線段長度
        p1 = np.array([x1, y1])
        p2 = np.array([x2, y2])
        line_point = np.array([new_nx * new_d, new_ny * new_d])
        line_normal = np.array([new_nx, new_ny])
        
        proj_p1 = p1 - (np.dot(p1 - line_point, line_normal)) * line_normal
        proj_p2 = p2 - (np.dot(p2 - line_point, line_normal)) * line_normal
        
        refine_line.append([[int(proj_p1[0]), int(proj_p1[1]), int(proj_p2[0]), int(proj_p2[1])]])

    return np.array(refine_line) if refine_line else None

def duplicate(lines):
    """
    移除幾何性質上重複或過於接近的線段

    Args:
        lines(np.array):待處理的線段

    Returns:
        np.array:刪除後的線段列表
    """
    check = [1] * len(lines) #判斷哪個線段要刪除
    for i in range(len(lines)):
        n1_x1, n1_y1, n1_x2, n1_y2 = lines[i][0]

        try:
            n1_nx = (n1_y2 - n1_y1) / np.sqrt((n1_y1 - n1_y2)**2 + (n1_x1 - n1_x2)**2)
            n1_ny = (n1_x1 - n1_x2) / np.sqrt((n1_y1 - n1_y2)**2 + (n1_x1 - n1_x2)**2)
            d1 = (n1_nx * n1_x1 + n1_ny * n1_y1)
        except ZeroDivisionError:
            continue

        for j in range(i + 1, len(lines)):
            n2_x1, n2_y1, n2_x2, n2_y2 = lines[j][0]

            try:
                n2_nx = (n2_y2 - n2_y1) / np.sqrt((n2_y1 - n2_y2)**2 + (n2_x1 - n2_x2)**2)
                n2_ny = (n2_x1 - n2_x2) / np.sqrt((n2_y1 - n2_y2)**2 + (n2_x1 - n2_x2)**2)
                d2 = (n2_nx * n2_x1 + n2_ny * n2_y1)
            except ZeroDivisionError:
                continue

            #判斷標準角度非常接近且距離也非常接近
            if (np.dot((n1_nx, n1_ny),(n2_nx, n2_ny)) > np.cos(np.deg2rad(0.75))) and (np.abs(d1 - d2) < 10):
                check[j] = 0 #標記為待刪除
    
    check_lines = [lines[i] for i in range(len(lines)) if check[i] == 1] #僅存下符合條件的線段
    return np.array(check_lines)

def classify_and_sort(lines):
    """
    將線段分為水平和垂直兩組，並分別進行排序

    Args:
        lines(np.array):待分類的線段列表

    Returns:
        tuple:已排序的水平線(list),已排序的垂直線(list)
    """
    horizontal_line = []
    vertical_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]

        angle = np.arctan2(y1-y2, x2-x1)
        angle_degree = np.abs(np.rad2deg(angle))

        # 角度在正負25度內的視為水平線
        if angle_degree < 25 or angle_degree > (180 - 25):
            horizontal_line.append(line)
        else :
            vertical_lines.append(line)

    #水平線：根據y座標中點由上至下排序
    horizontal_line.sort(key=lambda a: (a[0][1]+a[0][3])/2)
    #垂直線：根據x座標中點由左至右排序
    vertical_lines.sort(key=lambda a: (a[0][0]+a[0][2])/2)

    return horizontal_line, vertical_lines

def define_court_lines():
    """
    定義二維標準球場模型的幾何座標

    Returns:
        tuple:包含所有線的dictionary,水平線list,垂直線list
    """
    #二維標準球場模型
    court_model_lines = {
        #水平線順序為從上往下看 (y座標由小到大)
        "top1": [0, 0, 610, 0],
        "top2": [0, 76, 610, 76],
        "top3": [0, 468, 610, 468],
        "net": [0, 670, 610, 670],
        "bottom1": [0, 872, 610, 872],
        "bottom2": [0, 1264, 610, 1264],
        "bottom3": [0, 1340, 610, 1340],
        #垂直線順序為從左往右看 (x座標由小到大)
        "left1": [0, 0, 0, 1340],
        "left2": [203, 0, 203, 1340],
        "right1": [406, 0, 406, 1340],
        "right2": [610, 0, 610, 1340]
    }

    #二維標準球場水平線
    model_horizontal_lines = [
        court_model_lines["top1"],
        court_model_lines["top2"],
        court_model_lines["top3"],
        court_model_lines["net"],
        court_model_lines["bottom1"],
        court_model_lines["bottom2"],
        court_model_lines["bottom3"],
    ]

    #二維標準球場垂直線
    model_vertical_lines = [
        court_model_lines["left1"],
        court_model_lines["left2"],
        court_model_lines["right1"],
        court_model_lines["right2"],
    ]

    return court_model_lines, model_horizontal_lines, model_vertical_lines

def find_corner(h_lines, v_lines):
    """
    從已排序的線段列表中直接推算四個角點

    Args:
        h_lines(list):已由上至下排序的水平線
        v_lines(list):已由左至右排序的垂直線

    Returns:
        np.array:四個角落的座標點
    """
    #步驟1:從水平線中找出球場的頂部與底部y座標
    y = [py for line in h_lines for py in (line[0][1], line[0][3])]
    y_min = min(y)
    y_max = max(y)

    #步驟2:從已排序的垂直線中，直接選取最左與最右的邊線
    left_v = v_lines[0][0] 
    right_v = v_lines[-1][0]

    #步驟3:計算左右邊線在頂部與底部的x座標。利用y座標來判斷哪個是頂點(y較小)，哪個是底點(y較大)
    if left_v[1] < left_v[3]:
        top_left_x = left_v[0]
        bottom_left_x = left_v[2]
    else:
        top_left_x = left_v[2]
        bottom_left_x = left_v[0]

    if right_v[1] < right_v[3]:
        top_right_x = right_v[0]
        bottom_right_x = right_v[2]
    else:
        top_right_x = right_v[2]
        bottom_right_x = right_v[0]

    #步驟4:四個角點組成
    corner = np.array([
        [top_left_x, y_min], #左上角
        [top_right_x, y_min], #右上角
        [bottom_left_x, y_max], #左下角
        [bottom_right_x, y_max], #右下角
    ], dtype = np.float32)

    return corner

def find_H(corner, model_lines):
    """
    根據影片偵測到的角點與模型角點，計算Homography矩陣

    Args:
        corner(np.array):影片偵測的四個角點
        model_lines(dict):標準場地的字典

    Returns:
        np.array:四個角落的座標點
    """
    #標準場地的四個角點，並對應corner角點順序
    model_left_top = (model_lines['top1'][0], model_lines['left1'][1])
    model_left_bottom = (model_lines['bottom3'][0], model_lines['left1'][3])
    model_right_top = (model_lines['top1'][2], model_lines['right2'][1])
    model_right_bottom = (model_lines['bottom3'][2], model_lines['right2'][3])

    model_corners = np.array([
        model_left_top,
        model_right_top,
        model_left_bottom,
        model_right_bottom
    ], dtype=np.float32)

    H, mask = cv2.findHomography(model_corners, corner)

    return H

def yolo_detect_player(frame, yolo):
    """
    執行YOLOv8模型，偵測選手位置

    Args:
        frame(np.array):待偵測的影像
        yolo(YOLO):YOLO模型

    Returns:
        np.array:選手左上及右下座標點
    """
    result = yolo(frame, classes = [1], conf = 0.6)
    player = result[0]

    cord_xyxy = []
    for box in player.boxes:
        cord = box.xyxy[0].tolist()
        x1, y1, x2, y2 = map(int, cord)
        cord_xyxy.append([x1,y1,x2,y2])
    
    return cord_xyxy

def compute_player_position(cord):
    """
    從Bounding Box計算選手的腳下中心點

    Args:
        cord(list):選手左上及右下座標

    Returns:
        tuple:(x,y)座標代表選手腳下中心點的像素座標
    """
    x1, y1, x2, y2 = cord
    player_x = x1 + abs(x2-x1) / 2
    player_y = y2

    return player_x, player_y

def area_detect(x, y, model_lines):
    """
    根據選手的標準化球場座標 (X, Y)，判斷其所在的九宮格區域

    此函式將球場依據發球線劃分為九個區域 (0-8)，並區分上下半場的選手
    其中左右邊界的劃分線為 'left2' 和 'right1'

    Args:
        x(float):選手在標準場地上的x座標
        y(float):選手在標準場地上的y座標
        model_lines(dict):標準場地

    Returns:
        tuple:選手編號1或2,所在區域0-8,-1表示無效區域
    """
    player_number = 0
    area = -1
    #首先，判斷選手位於下半場(Player 1)或上半場(Player 2)
    if y > model_lines['net'][1]:
        player_number = 1 
        #根據y座標判斷前、中、後場
        if y > model_lines['net'][1] and y <= model_lines['bottom1'][1]: #前場
            if x < model_lines['left2'][0]:
                area = 0 #宮格1
            elif x >= model_lines['left2'][0] and x <= model_lines['right1'][0]:
                area = 1 #宮格2
            elif x > model_lines['right1'][0]:
                area = 2 #宮格3
        elif y > model_lines['bottom1'][1] and y <= model_lines['bottom2'][1]: #中場
            if x < model_lines['left2'][0]:
                area = 3 #宮格4
            elif x >= model_lines['left2'][0] and x <= model_lines['right1'][0]:
                area = 4 #宮格5
            elif x > model_lines['right1'][0]:
                area = 5 #宮格6
        elif y >= model_lines['bottom2'][1]: #後場
            if x < model_lines['left2'][0]:
                area = 6 #宮格7
            elif x >= model_lines['left2'][0] and x <= model_lines['right1'][0]:
                area = 7 #宮格8
            elif x > model_lines['right1'][0]:
                area = 8 #宮格9
    #選手二
    else:
        player_number = 2 
        #選手二的區域與選手一呈現鏡像對稱
        if y >= model_lines['top3'][1] and y <= model_lines['net'][1]:
            if x > model_lines['right1'][0]:
                area = 0 #宮格1
            elif x >= model_lines['left2'][0] and x <= model_lines['right1'][0]:
                area = 1 #宮格2
            elif x < model_lines['left2'][0]:
                area = 2 #宮格3
        elif y > model_lines['top2'][1] and y < model_lines['top3'][1]:
            if x > model_lines['right1'][0]:
                area = 3 #宮格4
            elif x >= model_lines['left2'][0] and x <= model_lines['right1'][0]:
                area = 4 #宮格5
            elif x < model_lines['left2'][0]:
                area = 5 #宮格6
        elif y <= model_lines['top2'][1]:
            if x > model_lines['right1'][0]:
                area = 6 #宮格7
            elif x >= model_lines['left2'][0] and x <= model_lines['right1'][0]:
                area = 7 #宮格8
            elif x < model_lines['left2'][0]:
                area = 8 #宮格9

    return player_number, area

def from_detect_to_model(H, cord_x, cord_y):
    """
    使用Homography逆矩陣，將影像像素座標轉換為標準場地座標

    Args:
        H(np.array):Homography矩陣
        cord_x(float):選手在影像上的 x 像素座標
        cord_y(float):選手在影像上的 y 像素座標

    Returns:
        tuple:選手在標準球場上的(x,y)座標
    """
    #H是從模型到影像的映射，所以需要逆矩陣來進行反向映射
    inverse_H = np.linalg.inv(H)
    p = np.array([[[cord_x, cord_y]]], dtype=np.float32)
    tran_p = cv2.perspectiveTransform(p, inverse_H)

    return tran_p[0][0][0], tran_p[0][0][1]

def status_panel(p1_left_count, p1_middle_count, p1_right_count, p2_left_count, p2_middle_count, p2_right_count, p1_current, p2_current, video_w):
    """
    建立一個用於顯示即時分析數據的顯示面板

    Args:
        p1_left_count(int):選手一左邊區域的資訊
        p1_middle_count(int):選手一中間區域的資訊
        p1_right_count(int):選手一右邊區域的資訊
        p2_left_count(int):選手二左邊區域的資訊
        p2_middle_count(int):選手二中間區域的資訊
        p2_right_count(int):選手二右邊區域的資訊
        p1_current(int):選手一目前的所在位置
        p2_current(int):選手二目前所在的位置
        video_w(int):影片寬度，用於決定面板寬度

    Returns:
        np.array:面板影像
    """
    panel = np.zeros((280, video_w, 3), dtype=np.uint8)

    cv2.putText(panel, "Player 1", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 191, 1), 2)
    cv2.putText(panel, "Player 2", (video_w//2, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 191, 1), 2)

    p1_area_text = f"Current area: {p1_current}"
    p2_area_text = f"Current area: {p2_current}"

    p1_left_counts_text = f"Left: {p1_left_count}"
    p2_left_counts_text = f"Left: {p2_left_count}"
    p1_middle_counts_text = f"Middle: {p1_middle_count}"
    p2_middle_counts_text = f"Middle: {p2_middle_count}"
    p1_right_counts_text = f"Right: {p1_right_count}"
    p2_right_counts_text = f"Right: {p2_right_count}"

    cv2.putText(panel, p1_area_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(panel, p2_area_text, (video_w//2, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.putText(panel, p1_left_counts_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(panel, p2_left_counts_text, (video_w//2, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.putText(panel, p1_middle_counts_text, (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(panel, p2_middle_counts_text, (video_w//2, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.putText(panel, p1_right_counts_text, (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(panel, p2_right_counts_text, (video_w//2, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return panel


#====================================================================================
#主程式執行
#====================================================================================
if __name__ == '__main__':
    #一、載入影片與 YOLO 模型，並取得影片第一幀
    video = cv2.VideoCapture(r"D:\yolov8_project\final_video\001.mp4") 
    yolo = YOLO(r"D:\yolov8_project\runs\detect\002\weights\best.pt")
    ret, origin_frame = video.read()
    
    #二、場地偵測與計算H矩陣
    #2.1 提取白色像素點
    white_frame = white_pixel_extraction(origin_frame, sigma_l = 120, sigma_d = 50, tau = 8)
    
    #2.2 機率霍夫變換、精煉予刪除重複線段
    PHT_frame, PHT_lines = PHT(white_frame, origin_frame, T = 150, minLength = 20, LineGap = 120)
    final_lines = PHT_lines
    for i in range(3):
        final_lines = refine(final_lines, white_frame, sigma_r = 5)
        final_lines = duplicate(final_lines)

     #2.3 計算 Homography 矩陣
    model_lines, model_horizontal_lines, model_vertical_lines = define_court_lines()
    candidate_horizontal_lines, candidate_vertical_lines = classify_and_sort(final_lines)

    corner = find_corner(candidate_horizontal_lines, candidate_vertical_lines)
    H = find_H(corner, model_lines)
    if H is None:
        print("錯誤：Homography矩陣計算失敗")
        exit()
    
    #三、偵測影片，將YOLO模型與H矩陣做結合
    #3.1 初始化計數器與狀態變數 
    player1_count = 9 * [0]
    player2_count = 9 * [0]
    player1_last_area = -1
    player2_last_area = -1
    #3.2 主迴圈執行
    while True:
        ret, frame = video.read()
        if ret is False:
            print("The video has finished playing or there is a loading error.")
            break

        player_xyxy = yolo_detect_player(frame, yolo)
        for cord in player_xyxy:
            x1, y1, x2, y2 = cord
            player_x, player_y = compute_player_position(cord)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.circle(frame, (int(player_x), int(player_y)), 8, (0, 0, 255), -1)

            tran_x, tran_y = from_detect_to_model(H, player_x, player_y)
            player_number, area = area_detect(tran_x, tran_y, model_lines)
            if player_number == 1 and area != -1 and player1_last_area != area:
                player1_count[area] += 1
                player1_last_area = area
            elif player_number == 2  and area != -1 and player2_last_area != area:
                player2_count[area] += 1
                player2_last_area = area

        for i in model_lines:
            p1 = np.array([[[model_lines[i][0], model_lines[i][1]]]], dtype=np.float32)
            p2 = np.array([[[model_lines[i][2], model_lines[i][3]]]], dtype=np.float32)
                
            proj_p1 = cv2.perspectiveTransform(p1, H)[0][0]
            proj_p2 = cv2.perspectiveTransform(p2, H)[0][0]

            pt1 = (int(proj_p1[0]), int(proj_p1[1]))
            pt2 = (int(proj_p2[0]), int(proj_p2[1]))
            if  i == 'net':
                cv2.line(frame, pt1, pt2, (0, 0, 0), 2)
            else:
                cv2.line(frame, pt1, pt2, (0, 0, 255), 2)   
        
        panel = status_panel(
            [player1_count[0], player1_count[3], player1_count[6]],
            [player1_count[1], player1_count[4], player1_count[7]],
            [player1_count[2], player1_count[5], player1_count[8]],
            [player2_count[0], player2_count[3], player2_count[6]],
            [player2_count[1], player2_count[4], player2_count[7]],
            [player2_count[2], player2_count[5], player2_count[8]],
            player1_last_area+1,
            player2_last_area+1, 
            frame.shape[1])
        
        frame = np.vstack((frame,panel))

        #更改尺寸，影片原為1920*1080
        new_width = int(frame.shape[1] * 70 / 100)
        new_height = int(frame.shape[0] * 70 / 100)
        frame = cv2.resize(frame, (new_width, new_height))
        cv2.imshow("Badminton Analysis System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  

    cv2.waitKey(0)
    video.release()
    cv2.destroyAllWindows()
