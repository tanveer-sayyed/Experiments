#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 15:48:02 2020

@author: tanveer
"""
import cv2

def sketch(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray_blur = cv2.GaussianBlur(img_gray, (5,5), 0) # clean up the noise in the image
    canny_edges = cv2.Canny(img_gray_blur, 20, 60)
    ret, mask = cv2.threshold(canny_edges, 60, 255, cv2.THRESH_BINARY_INV)
    return mask

if __name__ == "__main__":
    capture = cv2.VideoCapture(0)
    while True:
        ret, frame = capture.read()
        cv2.imshow("LIVE sketch... ", sketch(frame))
        if cv2.waitKey(1) == 13: # 13 is the Enter key
            break
    capture.release()
    cv2.destroyAllWindows()
