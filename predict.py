import cv2
import numpy as np
from imageProcess import extract, scale_and_centre, order_corner_points
from tensorflow.python.keras.models import load_model
import tensorflow as tf

def display_image(img):
    cv2.imshow('image', img)  # Display the image
    cv2.waitKey(0)  # Wait for any key to be pressed (with the image window active)
    cv2.destroyAllWindows()  # Close all windows

def extract_number_image(img_grid):
    tmp_sudoku = [[0 for i in range(9)] for j in range(9)]
    for i in range(9):
        for j in range(9):

            image = img_grid[i][j]
            image = cv2.resize(image, (28, 28))
            original = image.copy()

            thresh = 128  # define a threshold, 128 is the middle of black and white in grey scale
            # threshold the image
            gray = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)[1]
            #Let (x,y) be the top-left coordinate of the rectangle and (w,h) be its width and height.
            # Find contours
            cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]

            for c in cnts:
                x, y, w, h = cv2.boundingRect(c)

                if (x < 3 or y < 3 or h < 3 or w < 3):
                    # Note the number is always placed in the center
                    # Since image is 28x28
                    # the number will be in the center thus x >3 and y>3
                    # Additionally any of the external lines of the sudoku will not be thicker than 3
                    continue
                ROI = gray[y:y + h, x:x + w]
                ROI = scale_and_centre(ROI, 120)
                # display_image(ROI)

                # Writing the cleaned cells
                cv2.imwrite("CleanedBoardCells/cell{}{}.png".format(i, j), ROI)
                tmp_sudoku[i][j] = predict(ROI)

    return tmp_sudoku

def predict(img_grid):
    image = img_grid.copy()
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # image = cv2.threshold(image, 140, 255, cv2.THRESH_BINARY)[1]
    image = cv2.resize(image, (28, 28))
    # display_image(image)
    image = image.astype('float32')
    image = image.reshape(1, 28, 28, 1)
    image /= 255

    # plt.imshow(image.reshape(28, 28), cmap='Greys')
    # plt.show()
    model = load_model('classification_complete.h5')
    pred = model.predict(image.reshape(1, 28, 28, 1), batch_size=1)
    # pred = model.predict(image.reshape(1, 28*28), batch_size=1)

    # print(pred.argmax())

    return pred.argmax()

def possible(y,x,n): #y=row,x=col
	global cells
	for i in range(9):
		if cells[y][i]==n:
			return False

	for i in range(9):
		if cells[i][x]==n:
			return False
	l=(y//3)*3
	m=(x//3)*3
	for i in range(3):
		for j in range(3):
			if cells[l+i][m+j]==n:
				return False

	return True

def solve():
	global cells
	# print(grid,end='\n\n')
	for y in range(9):
		for x in range(9):
			if cells[y][x]==0:
				for n in range(1,10):
					if possible(y,x,n):
						cells[y][x]=n
						if solve():
							return True
						else:
							cells[y][x]=0
				return False
	return True 

#--------------------------------------------    

physical_devices = tf.config.experimental.list_physical_devices('GPU')

if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


cells=np.array(extract_number_image(extract()))

print(cells)
solve()
print(cells)


