import cv2
import numpy as np
import matplotlib.pyplot as plt

def colorHistogram(img, cspace='RGB', binsize=128):
   # Add you code here
    #ANOTHER SOLUTION
    #cspaceDict = {"HSV": cv2.COLOR_BGR2HSV,"RGB": cv2.COLOR_BGR2RGB,"XYZ": cv2.COLOR_BGR2XYZ,"YCBCR": cv2.COLOR_BGR2YCR_CB }
    #imgSplit = cv2.cvtColor(img,cspaceDict[cspace])
    #a, b, c = imgSplit[:,:,0], imgSplit[:,:,1], imgSplit[:,:,2]
    if cspace == 'CrCb':
        rf = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    elif cspace == 'XYZ':
        rf = cv2.cvtColor(img, cv2.COLOR_BGR2XYZ)
    elif cspace == 'Lab':
        rf = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    elif cspace == 'HSV':
        rf = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    else :
        rf = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    one, two, three = rf[:,:,0], rf[:,:,1], rf[:,:,2]
    one_hist, one_bin = np.histogram(one, binsize, density=True)
    two_hist, two_bin = np.histogram(two, binsize, density=True)
    three_hist, three_bin = np.histogram(three, binsize, density=True)
    oneTwoThree_hist = np.concatenate((one_hist, two_hist, three_hist))
    return oneTwoThree_hist

input_dir = 'dataset/test/03.bmp'
img = cv2.imread(input_dir)
input_dir2 = 'dataset/groundtruth/03.bmp'
img2 = cv2.imread(input_dir2)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
testing = img.copy()

kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(4,4)) #MORPH_RECT,MORPH_CROSS,MORPH_ELLIPSE 

# Perform erosion and dilation separately and then subtract
img = cv2.erode(img, kernel, iterations = 5) #current best 3,3
img = cv2.dilate(img, kernel, iterations = 5)
#img_hist = colorHistogram(img)



#pos = np.arange(384)
#plt.bar(pos, img_hist), plt.title('RGB Color histogram')
#plt.show()

lower = 0
upper = 50

lower2 = 80
upper2 = 200

mask = cv2.inRange(img, lower, upper)
mask2 = cv2.inRange(img, lower2, upper2)

outcome = mask2 + mask
res = cv2.bitwise_and(mask, img, mask=mask)

plt.subplot(431),plt.imshow(testing,cmap ='gray')
plt.title('original')
plt.subplot(432),plt.imshow(img,cmap ='gray')
plt.title('testing')
plt.subplot(433),plt.imshow(outcome,cmap ='gray')
plt.title('mask')
plt.show()


outcome = cv2.erode(outcome, kernel, iterations = 5)
numLabels, output_labels1 = cv2.connectedComponents(outcome, 4, cv2.CV_32S)
print(numLabels)


#plt.imshow(mask , cmap = 'gray')
#plt.show()


plt.subplot(431),plt.imshow(testing,cmap ='gray')
plt.title('original')
plt.subplot(432),plt.imshow(img2)
plt.title('answer')
plt.subplot(433),plt.imshow(output_labels1, cmap = 'jet')
plt.title('answer'), plt.xticks([]), plt.yticks([])
plt.show()
