
import cv2
import numpy as np
import scipy.spatial

def display_image(img):
    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def processing(img,skip_dilate=False):
    # img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    process=cv2.GaussianBlur(img.copy(),(9,9),0)

    process= cv2.adaptiveThreshold(process, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    process=cv2.bitwise_not(process,process)

    if not skip_dilate:
        # This is only used for sudoku processing and not for cell processing
        # np.uint8 will wrap.
        # For example, 235+30 = 9.
        kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], np.uint8)
        process = cv2.dilate(process, kernel)
        # print(process.dtype)

    return process

def find_contours(img):
    ext_contours=cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ext_contours= ext_contours[0] if len(ext_contours)==2 else ext_contours[1]
    ext_contours = sorted(ext_contours, key=cv2.contourArea, reverse=True)
    
    for c in ext_contours:
        peri=cv2.arcLength(c, True)

        approx= cv2.approxPolyDP(c, 0.015*peri, True)

        if len(approx)==4:
            return approx

def order_corner_points(corners):
    corners=[(corner[0][0],corner[0][1]) for corner in corners]


    corners=np.array(corners)
    # Order along X axis
    Xorder = corners[np.argsort(corners[:, 0]), :]

    left = Xorder[:2, :]
    right = Xorder[2:, :]

    # Order along Y axis
    left = left[np.argsort(left[:, 1]), :]
    (tl, bl) = left

    # use distance to get bottom right
    D = scipy.spatial.distance.cdist(tl[np.newaxis], right, "euclidean")[0]
    (br, tr) = right[np.argsort(D)[::-1], :]

    return np.array([tl, tr, br, bl])

def perspective_transform(image,corners):
    
    
    
    
    ordered_corners = order_corner_points(corners)
    
    top_l, top_r, bottom_r, bottom_l = ordered_corners[0], ordered_corners[1], ordered_corners[2], ordered_corners[3]

    width_A = np.sqrt(((bottom_r[0] - bottom_l[0]) ** 2) + ((bottom_r[1] - bottom_l[1]) ** 2))
    width_B = np.sqrt(((top_r[0] - top_l[0]) ** 2) + ((top_r[1] - top_l[1]) ** 2))
    width = max(int(width_A), int(width_B))

    # Determine height of new image which is the max distance between
    # (top right and bottom right) or (top left and bottom left) y-coordinates
    height_A = np.sqrt(((top_r[0] - bottom_r[0]) ** 2) + ((top_r[1] - bottom_r[1]) ** 2))
    height_B = np.sqrt(((top_l[0] - bottom_l[0]) ** 2) + ((top_l[1] - bottom_l[1]) ** 2))
    height = max(int(height_A), int(height_B))

    dimensions = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1],
                           [0, height - 1]], dtype="float32")

    # Convert to Numpy format
    ordered_corners = np.array(ordered_corners, dtype="float32")

    # calculate the perspective transform matrix and warp
    # the perspective to grab the screen
    grid = cv2.getPerspectiveTransform(ordered_corners, dimensions)

    # Return the transformed image
    
    return cv2.warpPerspective(image, grid, (width, height))

def create_image_grid(img):
    grid=np.copy(img)
    edge_h=grid.shape[0]
    edge_w=grid.shape[1]
    celledge_width=edge_w // 9
    celledge_height=edge_h // 9

    grid=cv2.bitwise_not(grid,grid) 
 

    tempgrid=[]
    for i in range(celledge_height,edge_h+1,celledge_height): #+1 taken as last range is not printed in [a:b] printed till b-1
        for j in range(celledge_width,edge_w+1,celledge_width):

            rows=grid[i-celledge_height:i]
            tempgrid.append([rows[k][j - celledge_width:j] for k in range(len(rows))])
    # print(len(tempgrid))

    finalgrid = []   #all pixels stored for each cell
    for i in range(0, len(tempgrid) - 8, 9):
        finalgrid.append(tempgrid[i:i + 9])
    
    # print(len(finalgrid))
    for i in range(9):
        for j in range(9):
            finalgrid[i][j] = np.array(finalgrid[i][j])

    try:
        for i in range(9):
            for j in range(9):
                np.os.remove("Board_Cells/cell" + str(i) + str(j) + ".jpg")
    except:
        pass
    for i in range(9):
        for j in range(9):
            cv2.imwrite(str("Board_Cells/cell" + str(i) + str(j) + ".jpg"), finalgrid[i][j])

    return finalgrid
    
def scale_and_centre(img, size, margin=20, background=0):
    """Scales and centres an image onto a new background square."""
    h, w = img.shape[:2]

    def centre_pad(length):
        """Handles centering for a given length that may be odd or even."""
        if length % 2 == 0:
            side1 = int((size - length) / 2)
            side2 = side1
        else:
            side1 = int((size - length) / 2)
            side2 = side1 + 1
        return side1, side2

    def scale(r, x):
        return int(r * x)

    if h > w:
        t_pad = int(margin / 2)
        b_pad = t_pad
        ratio = (size - margin) / h
        w, h = scale(ratio, w), scale(ratio, h)
        l_pad, r_pad = centre_pad(w)
    else:
        l_pad = int(margin / 2)
        r_pad = l_pad
        ratio = (size - margin) / w
        w, h = scale(ratio, w), scale(ratio, h)
        t_pad, b_pad = centre_pad(h)

    img = cv2.resize(img, (w, h))
    img = cv2.copyMakeBorder(img, t_pad, b_pad, l_pad, r_pad, cv2.BORDER_CONSTANT, None, background)
    return img     



def extract():
    img=cv2.imread('sudoku_1.jpg',0)
  
    print(img.shape)
    processed_sudoku=processing(img)

    sudoku_corners=find_contours(processed_sudoku)
    
    perspective=perspective_transform(img, sudoku_corners)
    display_image(perspective)
    perspective=cv2.resize(perspective,(450,450))
    grid=create_image_grid(perspective)
    
    
    return grid

if __name__=="__main__":
    extract()
    while True:
        key=cv2.waitKey(0)
        if key:
            break

    cv2.destroyAllWindows()