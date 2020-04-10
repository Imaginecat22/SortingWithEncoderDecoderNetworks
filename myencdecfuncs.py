#this is a python module written by Robert Walters
#for the purposes of generating and managing datasets related to encoder-decoder neural network testing

import numpy as np
import keras

#create training data for simple row-by-row sorting
def createRowSortTrainingData(vectors, vlength, startval, endval, padval):
	unsorted = np.random.randint(startval, endval, size=(vectors, vlength), dtype=int)
	padvec = np.full((vectors, 1), padval, dtype=int)
	vsorted = np.sort(unsorted, axis=1)
	vshifted = vsorted[:,:-1]
	padded = np.hstack((padvec, vshifted))
	return unsorted, padded, vsorted

#flips part of a 1D numpy array
def nppartflip(inputarr, lstart, lend):
	inputarr[lstart:lend] = inputarr[lstart:lend][::-1]
	return inputarr

#flips part of each vector in a 2D numpy matrix
def npPartialFlip2D(inarr2D, lowval, highval):
	temp = []
	for vector in inarr2D:
		cnt = 0
		for num in vector:
			if(num < highval and num >=lowval):
				cnt += 1
		nvector = nppartflip(vector, 0, cnt)
		temp.append(nvector)
	outarr2D = np.array(temp)
	return outarr2D

#create training data for the LNS task; where 0-9 are "numbers" and 10-36 are "letters"
def createLNSTrainingData(vectors, vlength, startval, endval, padval):
	unsorted = np.random.randint(startval, endval, size=(vectors, vlength), dtype=int)
	padvec = np.full((vectors, 1), padval, dtype=int)
	vsorted = np.sort(unsorted, axis=1)
	#instead of just sorting the data, I need to create two groups: "number" data and "letter" data
	#the number data will be sorted in descending (highest to lowest) order
	#the letter data will be sorted alphabetically, as expected
	#since the NN cannot take in numbers, we'll say the digits 0-9 are numbers
	#and numbers 10-36 (26 total) are "letters"
	vflipped = npPartialFlip2D(vsorted, 0, 9)	
	vshifted = vflipped[:,:-1]
	padded = np.hstack((padvec, vshifted))
	return unsorted, padded, vflipped 


#flattens each 2D slice of a 3D numpy matrix
def matFlat(array3D, num_matrices, display=0):
	mlist = []
	for m in range(len(array3D)):
		matrix = array3D[m]
		if display == 1:
			print("before ", matrix.shape)
		#matrix.flatten()	
		flat = np.hstack(matrix)	
		if display == 1:
			print("after ", flat.shape)
		mlist.append(flat)
	mout = np.array(mlist)
	return mout

#this is more what I had in mind
#sorts the matrix by treating each array as an object and sorting in ascending order
#based on the FIRST value in that matrix (kinda sorting the first column?)
def vecsort(matrix):
	sortdict = {}
	dupcnt = 1
	for m in range(len(matrix)):
		key = matrix[m][0]
		value = matrix[m][1:]
		if key not in sortdict:
			sortdict[key] = value
		else:
			key = key + (100 * dupcnt)
			sortdict[key] = value
			dupcnt += 1
	arrlist = []	
	for k in sorted (sortdict):
		if k < 100:
			array = np.append(k, sortdict[k])
			arrlist.append(array)
		else:
			dup = k
			dupcnt -= 1
			while dup > 100:
				dup -= 100
			#print("dup: ", dup)
			array = np.append(dup, sortdict[k])
			length = len(arrlist)
			for c in range(length):
				if dup == arrlist[c][0]:
					arrlist.insert(c, array)
					break
			
				
		#print(k, sortdict[k], "\n\n")
	if (dupcnt != 1):
		print("dupcnt: ", dupcnt)
	temp = len(arrlist)	
	newmatrix = np.array(arrlist)	
	return newmatrix


#sorts the first column, then flattens the 2D matrix to a 1D array
def colSortFlat(unsorted): 
	sorted = []
	for matrix in range(len(unsorted)):
		tmatrix = unsorted[matrix]
		#rotate = np.rot90(tmatrix, -1)
		#rotate[0, :] = np.sort(rotate[0, :])
		#rotate[0, :] = np.flip(rotate[0, :])
		#vsorted = np.rot90(rotate, 1)
		vsorted = vecsort(tmatrix)
		sorted.append(vsorted)
	sortout = np.array(sorted)
	flattened = matFlat(sortout, len(sortout))
	return flattened

#sorts the first column of each 2D matrix in ascending order (does not flatten)
def colSort(unsorted): 
	sorted = []
	#now proceed as normal
	for matrix in range(len(unsorted)):
		tmatrix = unsorted[matrix]
		#rotate clockwise
		rotate = np.rot90(tmatrix, -1)
		#sort, then reverse the first "row"
		rotate[0, :] = np.sort(rotate[0, :])
		rotate[0, :] = np.flip(rotate[0, :])
		#rotate it back, now first column is sorted
		vsorted = np.rot90(rotate, 1)
		sorted.append(vsorted)
	sortout = np.array(sorted)
	return sortout


#adds the start token to the start of each 1D flattened array to prep for teacher forcing
def colPadFlat(flatsorted, padval):
	flatvec = flatsorted #matFlat(flatsorted, len(flatsorted))
	#len(flatvec)
	print("flatvec: ", flatvec.shape, len(flatvec))
	padvec = np.full((len(flatvec), 1), padval, dtype=int)
	vshifted = flatvec[:,:-1]
	#print(flatvec[0], "\n\n", flatvec[1])
	padded = np.hstack((padvec, vshifted))
	return padded


#adds the start token to each 2D matrix to prep for teacher forcing
#DOES NOT WORK; ACTUALLY ADDS AN EXTRA COLUMN
def colPad(sorted, padval, vcols):
	padvec = np.full((1, vcols), padval, dtype=int) 
	padded = []
	for matrix in range(len(sorted)):
		tmatrix = sorted[matrix]
		vshifted = tmatrix[1:] #tmatrix[:,:-1]
		vpadded = np.vstack((padvec, vshifted))
		padded.append(vpadded)
	padout = np.array(padded)
	return padout

#the old columnsort function. Sorts the first column of each 10xN matrix (sample size)
#When put into the network, it looks for patters in every row. The pattern only exists in columns, so it 
#would never be able to learn it
def tempfunc(unsorted, size):
	buffer = np.copy(unsorted)
	rotate = np.rot90(buffer, -1)
	#now I need to sort the first "row" piecewise for every 10 numbers	
	r = size/10 #VECTORS MUST BE EVENLY DIVISIBLE BY 10!!!!!!
	st = 0
	ra = st + 10
	for x in range(r):
		chunk = rotate[0,st:ra] #grab a piece of the first row
		schunk = np.sort(chunk) #sort that piece
		fchunk = np.flip(schunk) #flip that sorted piece
		rotate[st:ra] = fchunk #put the sorted/flipped piece back in place
		st += 10
		ra += 10
	return rotate


#create training data where the first column of a vrowsXvcols matrix is sorted, then flattened into a series of
#1D arrays (making a 2D matrix, with each row becoming vrowsXvcols in length)
def createColumnTrainingData(num_matrices, vrows, vcols, startval, endval, padval):
	unsorted = np.random.randint(startval, endval, size=(num_matrices, vrows, vcols), dtype=int)
	vsorted = colSortFlat(unsorted)
	flattened = matFlat(unsorted, len(unsorted))
	padded = colPadFlat(vsorted, padval)
	return flattened, padded, vsorted


#decodes one-hot encoded sequences	
def OHdecode(sequence):
	return [np.argmax(vector) for vector in sequence]

#this one-hot encodes the training data, preparing it for use with the Enc-Dec Network
def prepTrainingData(unsorted, padded, target, totalClasses):
	in1 = keras.utils.to_categorical(unsorted, num_classes = totalClasses)
	in2 = keras.utils.to_categorical(padded, num_classes = totalClasses)
	out = keras.utils.to_categorical(target, num_classes = totalClasses)
	return in1, in2, out

#this creates and encodes the training data (using a group of default variables, if one wishes)
#for regular sort, input 1 | for LNS sort, input 2 | for (now flattened) first column sort, input 3
def quickBuild(vectors, funcselect, vlength=10, startval=1, endval=36, padval=0, totalClasses=37):
	if funcselect == 1:
		unsorted, padded, target = createRowSortTrainingData(vectors, vlength, startval, endval, padval)
	elif funcselect == 2:
		unsorted, padded, target = createLNSTrainingData(vectors, vlength, startval, endval, padval)
	elif funcselect == 3:
		unsorted, padded, target = createColumnTrainingData(vectors, vlength, vlength, 
			startval, endval, padval)
	else:
		print("Quick-Create Error: No function selected!")
	
	ret1, ret2, out = prepTrainingData(unsorted, padded, target, totalClasses)	
	return ret1, ret2, out


#this creates the training data WITHOUT ENCODING (using a group of default variables, if one wishes)
#for regular sort, input 1 | for LNS sort, input 2 | for (now flattened) first column sort, input 3
def quickCreate(vectors, funcselect, vlength=10, startval=1, endval=36, padval=0, totalClasses=37):
	if funcselect == 1:
		unsorted, padded, target = createRowSortTrainingData(vectors, vlength, startval, endval, padval)
	elif funcselect == 2:
		unsorted, padded, target = createLNSTrainingData(vectors, vlength, startval, endval, padval)
	elif funcselect == 3:
		unsorted, padded, target = createColumnTrainingData(vectors, vlength, vlength, 
			startval, endval, padval)
	else:
		print("Quick-Create Error: No function selected!")
	
	return unsorted, padded, target


#this is a testing function
#it creates and then prints a all types of training input data for testing (and demo) purposes
def quickTest(vectors=2, full=0):
	one, two, three = quickCreate(vectors, 1, vlength=10, endval=10, totalClasses=6)
	print(one, "\n\n", two, "\n\n", three)
	print("----------------")
	one, two, three = quickCreate(vectors, 2, vlength=10, endval=10, totalClasses=6)
	print(one, "\n\n", two, "\n\n", three)
	print("----------------")
	one, two, three = quickCreate(vectors, 3, vlength=10, endval=10, totalClasses=6)
	print(one, "\n\n", two, "\n\n", three)
	print("----------------")
	
	if full == 1:
		one, two, three = prepTrainingData(one, two, three, 6)	
	
	return one, two, three
	


