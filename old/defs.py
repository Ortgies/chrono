from imutils.object_detection import non_max_suppression
import numpy as np
import pytesseract
import argparse
import cv2
import os
from matplotlib import pyplot as plt
from os import listdir
import multiprocess as mp
import inspect
from nltk.metrics.distance import edit_distance
from sklearn.model_selection import KFold
from functools import partial


# define the two output layer names for the EAST detector model that
# we are interested in -- the first is the output probabilities and the
# second can be used to derive the bounding box coordinates of text
layerNames = [
	"feature_fusion/Conv_7/Sigmoid",
	"feature_fusion/concat_3"]
# load the pre-trained EAST text detector
print("[INFO] loading EAST text detector...")
net = cv2.dnn.readNet('frozen_east_text_detection.pb')



def decode_predictions(scores, geometry):
	# grab the number of rows and columns from the scores volume, then
	# initialize our set of bounding box rectangles and corresponding
	# confidence scores
	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []
	# loop over the number of rows
	for y in range(0, numRows):
		# extract the scores (probabilities), followed by the
		# geometrical data used to derive potential bounding box
		# coordinates that surround text
		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]
		# loop over the number of columns
		for x in range(0, numCols):
			# if our score does not have sufficient probability,
			# ignore it
			if scoresData[x] < 0.5: ## MIN CONFIDENCE TWEAK
				continue
			# compute the offset factor as our resulting feature
			# maps will be 4x smaller than the input image
			(offsetX, offsetY) = (x * 4.0, y * 4.0)
			# extract the rotation angle for the prediction and
			# then compute the sin and cosine
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)
			# use the geometry volume to derive the width and height
			# of the bounding box
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]
			# compute both the starting and ending (x, y)-coordinates
			# for the text prediction bounding box
			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)
			# add the bounding box coordinates and probability score
			# to our respective lists
			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])
	# return a tuple of the bounding boxes and associated confidences
	return (rects, confidences)


def get_text(path):

   # load the input image and grab the image dimensions
    image = cv2.imread(path)
    
    
    orig = image.copy()
    (origH, origW) = image.shape[:2]
    # set the new width and height and then determine the ratio in change
    # for both the width and height
    (newW, newH) = (640, 640)
    rW = origW / float(newW)
    rH = origH / float(newH)
    # resize the image and grab the new image dimensions
    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2] 
    
    # construct a blob from the image and then perform a forward pass of
    # the model to obtain the two output layer sets
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)
    # decode the predictions, then  apply non-maxima suppression to
    # suppress weak, overlapping bounding boxes
    (rects, confidences) = decode_predictions(scores, geometry)
    boxes = non_max_suppression(np.array(rects), probs=confidences)
    
    # initialize the list of results
    results = []
    # loop over the bounding boxes
    for (startX, startY, endX, endY) in boxes:
        # scale the bounding box coordinates based on the respective
        # ratios
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)
        # in order to obtain a better OCR of the text we can potentially
        # apply a bit of padding surrounding the bounding box -- here we
        # are computing the deltas in both the x and y directions
        dX = int((endX - startX) * 0.0)
        dY = int((endY - startY) * 0.0)
        # apply padding to each side of the bounding box, respectively
        startX = max(0, startX - dX)
        startY = max(0, startY - dY)
        endX = min(origW, endX + (dX * 2))
        endY = min(origH, endY + (dY * 2))
        # extract the actual padded ROI
        roi = orig[startY:endY, startX:endX]
    
        # in order to apply Tesseract v4 to OCR text we must supply
        # (1) a language, (2) an OEM flag of 4, indicating that the we
        # wish to use the LSTM neural net model for OCR, and finally
        # (3) an OEM value, in this case, 7 which implies that we are
        # treating the ROI as a single line of text
        config = ("-l eng --oem 1 --psm 7")
        text = pytesseract.image_to_string(roi, config=config)
        # add the bounding box coordinates and OCR'd text to the list
        # of results
        results.append(((startX, startY, endX, endY), text))
    
    return(results)

def get_word_list(path):
    print(path)
    i = path.split('.')[0]

    try:
        res = get_text('./images/{}'.format(path))
    except:
        print('Error')
        return []
    
    words = []
    for r in res:
        word = r[1].split('\n')[0]
        ws = word.split(' ')
        ws = [i for i in ws if (('.com' not in i) & ('24' not in i))]
        
        for w in ws:
            w = ''.join([i for i in w if i.isalpha()])
            
            if(len(w) > 2):
                words.append(w)
    return words



def sift4(s1, s2, max_offset=5):
    """
    This is an implementation of general Sift4.
    """
    t1, t2 = list(s1), list(s2)
    l1, l2 = len(t1), len(t2)
 
    if not s1:
        return l2
 
    if not s2:
        return l1
 
    # Cursors for each string
    c1, c2 = 0, 0
 
    # Largest common subsequence
    lcss = 0
 
    # Local common substring
    local_cs = 0
 
    # Number of transpositions ('ab' vs 'ba')
    trans = 0
 
    # Offset pair array, for computing the transpositions
    offsets = []
 
    while c1 < l1 and c2 < l2:
        if t1[c1] == t2[c2]:
            local_cs += 1
 
            # Check if current match is a transposition
            is_trans = False
            i = 0
            while i < len(offsets):
                ofs = offsets[i]
                if c1 <= ofs['c1'] or c2 <= ofs['c2']:
                    is_trans = abs(c2-c1) >= abs(ofs['c2'] - ofs['c1'])
                    if is_trans:
                        trans += 1
                    elif not ofs['trans']:
                        ofs['trans'] = True
                        trans += 1
                    break
                elif c1 > ofs['c2'] and c2 > ofs['c1']:
                    del offsets[i]
                else:
                    i += 1
            offsets.append({
                'c1': c1,
                'c2': c2,
                'trans': is_trans
            })
 
        else:
            lcss += local_cs
            local_cs = 0
            if c1 != c2:
                c1 = c2 = min(c1, c2)
 
            for i in range(max_offset):
                if c1 + i >= l1 and c2 + i >= l2:
                    break
                elif c1 + i < l1 and s1[c1+i] == s2[c2]:
                    c1 += i - 1
                    c2 -= 1
                    break
 
                elif c2 + i < l2 and s1[c1] == s2[c2 + i]:
                    c2 += i - 1
                    c1 -= 1
                    break
 
        c1 += 1
        c2 += 1
 
        if c1 >= l1 or c2 >= l2:
            lcss += local_cs
            local_cs = 0
            c1 = c2 = min(c1, c2)
 
    lcss += local_cs
    return round(max(l1, l2) - lcss + trans)


def get_string_distances(word, words):
    return -1*np.array([sift4(word, i, 20) for i in words])
    #return -1*np.array([edit_distance(word, i) for i in words])

