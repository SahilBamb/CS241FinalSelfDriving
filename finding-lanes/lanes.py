import cv2
import numpy as np
import matplotlib.pyplot as plt


def make_coordinates(image, line_parameters):
	slope, intercept = line_parameters
	y1 = image.shape[0]
	y2 = int(y1*(3/5))
	x1 = int((y1 - intercept)/slope)
	x2 = int((y2 - intercept)/slope)
	# print(image.shape)
	return np.array([x1, y1, x2, y2]) 

def average_slope_intercept(image, lines):
	left_fit = []
	right_fit = []
	if lines is not None:
		for line in lines:
			x1, y1, x2, y2 = line.reshape(4)
			#polyfit will process our points and give us the intercept and slope
			parameters = np.polyfit((x1,x2), (y1,y2), 1) 
			slope = parameters[0]
			intercept = parameters[1]
			if slope < 0:
				left_fit.append((slope, intercept))
			else:
				print(slope)
				right_fit.append((slope, intercept))
	left_fit_average = np.average(left_fit, axis=0)
	right_fit_average = np.average(right_fit, axis=0)
	# print(left_fit_average)
	# print(right_fit_average)
	try: 
		left_line = make_coordinates(image, left_fit_average)
		right_line = make_coordinates(image, right_fit_average)
		# print(left_fit_average, 'right')
		# print(right_fit_average, 'left')
		return np.array([left_line, right_line])
	except:
		return None


def canny(image):
	gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	blur = cv2.GaussianBlur(gray, (5,5), 0)
	canny = cv2.Canny(blur, 50, 150)
	return canny
	# cv2.imshow('result',canny)
	# cv2.waitKey(60*60)


def display_lines(image, lines):
	line_image = np.zeros_like(image)
	try:
		if (lines is not None):
			for line in lines:
				# print(line)
				if line is not None:
					x1, y1, x2, y2 = line.reshape(4)
					cv2.line(line_image, (x1,y1), (x2, y2), (255, 0, 0), 10)
		return line_image
	except:
		print('error occured',lines)
		return image

def region_of_interest(image):
	height = image.shape[0] #shape returns tuple of dimensions so this will return the rows
	polygons = np.array([
		[(200, height), (1100, height), (550, 250)]
		])
	# polygons = np.array([
	# 	[(200, height), (850, height), (550, 300), (460, 290)]
	# 	])
	mask = np.zeros_like(image)
	cv2.fillPoly(mask, polygons, 255) 
	masked_image = cv2.bitwise_and(image, mask)
	return masked_image


## Drawing lines for a static image ##

# image = cv2.imread('test_image.jpg')
# lane_image = np.copy(image)
# canny_image = canny(lane_image)
# cropped_image = region_of_interest(canny_image)
# lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
# averaged_lines = average_slope_intercept(lane_image, lines)
# line_image = display_lines(lane_image,averaged_lines)
# combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1) #second parameter is decreasing pixel intensity
# cv2.imshow("result", combo_image)
# cv2.waitKey(0)

# image = cv2.imread('test_image2.jpg')
# lane_image = np.copy(image)

# plt.imshow(lane_image)
# plt.show()

# exit()

## Drawing lines for video frames ##
cap = cv2.VideoCapture("SD2.mp4")
while (cap.isOpened()):
	_, frame = cap.read()
	canny_image = canny(frame)
	cropped_image = region_of_interest(canny_image)
	lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
	averaged_lines = average_slope_intercept(frame, lines)
	line_image = display_lines(frame,averaged_lines)
	combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1) #second parameter is decreasing pixel intensity
	cv2.imshow("result", combo_image)
	# if v2.waitKey(1)c == order('q'): 
	cv2.waitKey(2)