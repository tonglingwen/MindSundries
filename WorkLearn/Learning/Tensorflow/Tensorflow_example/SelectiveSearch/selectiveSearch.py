import sys
import cv2

if __name__ == '__main__':
	# If image path and f/q is not passed as command
	# line arguments, quit and display help message
	if len(sys.argv) < 3:
		print(__doc__)
		sys.exit(1)

	# speed-up using multithreads
	# 使用多线程加速
	cv2.setUseOptimized(True);
	cv2.setNumThreads(4);

	# read image
	im = cv2.imread(sys.argv[1])
	# resize image
	newHeight = 200
	newWidth = int(im.shape[1]*200/im.shape[0])
	im = cv2.resize(im, (newWidth, newHeight))    

	# create Selective Search Segmentation Object using default parameters
	# 使用默认参数创建选择性搜索分段对象
	ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

	# set input image on which we will run segmentation
	# 设置我们将运行分割的输入图像
	ss.setBaseImage(im)

	# Switch to fast but low recall Selective Search method
	# 切换到快速但低回调选择性搜索方法
	if (sys.argv[2] == 'f'):
		ss.switchToSelectiveSearchFast()

	# Switch to high recall but slow Selective Search method
	# 切换到高回调，但慢选择性搜索方法
	elif (sys.argv[2] == 'q'):
		ss.switchToSelectiveSearchQuality()
	# if argument is neither f nor q print help message
	# 如果参数既不是f也不是q打印帮助信息
	else:
		print(__doc__)
		sys.exit(1)

	# run selective search segmentation on input image
	# 在输入图像上运行选择性搜索分割
	rects = ss.process()
	print('Total Number of Region Proposals: {}'.format(len(rects)))

	# number of region proposals to show
	# 显示区域建议的数量
	numShowRects = 100
	# increment to increase/decrease total number
	# of reason proposals to be shown
	# 增加/减少要显示的理由建议的总数
	increment = 50

	while True:
		# create a copy of original image
		# 创建一个原始图像的副本
		imOut = im.copy()

		# itereate over all the region proposals
		# 遍历所有地区的建议
		for i, rect in enumerate(rects):
			# draw rectangle for region proposal till numShowRects
			# 为区域提议绘制矩形，直到numShowRects
			if (i < numShowRects):
				x, y, w, h = rect
				cv2.rectangle(imOut, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
			else:
				break

		# show output
		cv2.imshow("Output", imOut)

		# record key press
		k = cv2.waitKey(0) & 0xFF

		# m is pressed
		if k == 109:
			# increase total number of rectangles to show by increment
			# 增加矩形的总数以增量显示
			numShowRects += increment
		# l is pressed
		elif k == 108 and numShowRects > increment:
			# decrease total number of rectangles to show by increment
			# 减少总数的矩形显示增量
			numShowRects -= increment
		# q is pressed
		elif k == 113:
			break
	# close image show window
	cv2.destroyAllWindows()