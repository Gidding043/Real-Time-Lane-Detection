import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 1280) #CV_CAP_PROP_FRAME_WIDTH
cap.set(4, 720) #CV_CAP_PROP_FRAME_HEIGHT
cap.set(5, 0) #CV_CAP_PROP_FPS

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
            
    ignore_mask_color=255
    
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[255,0,0],thickness=5):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1,y1), (x2,y2), color, thickness)
            
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines=cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),
                         minLineLength=min_line_len,
                         maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1],3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

def weighted_img(img, initial_img, a, b, c):
    return cv2.addWeighted(initial_img, a, img, b, c)


rho = 2
theta = np.pi/180
threshold=90
min_line_len = 100
max_line_gap = 150

while cap.isOpened() :
    ret, frame = cap.read() # 캠 이미지 불러오기
    if not ret:
        break
    #print(frame.shape)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kernel_size = 5
    blur_frame = cv2.GaussianBlur(gray_frame, (kernel_size, kernel_size), 0)

    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blur_frame, low_threshold, high_threshold)
    mask = np.zeros_like(edges)

    ignore_mask_color = 255

    vertices=np.array([[(300, 720),
                   (330, 180),
                   (950, 180),
                   (980, 720)]], dtype=np.int32)
    
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    mask = region_of_interest(edges, vertices)
    lines = hough_lines(mask, rho, theta, threshold, 
                   min_line_len, max_line_gap)
    
    lines_edges=weighted_img(lines, frame, a=0.8, b=1.0, c=0.0)
    
    cv2.imshow('Lane Detection', lines_edges) # 불러온 이미지 출력하기
    if cv2.waitKey(1) == 27:
        break # esc to quit

cap.release()
cv2.destroyAllWindows()