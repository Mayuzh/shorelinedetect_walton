import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as tutils

import imageio

import cv2


def mask_to_uv(mask):
    # output pntU (width) and pntV (height)
    pntU = np.where(mask)[1] + 0.5
    pntV = np.where(mask)[0] + 0.5
    return pntU, pntV

################################################################################
################################################################################

def torch2plt(img):
    # extract numpy array and reshape for plotting
    npimg = img.numpy()
    return np.transpose(npimg, (1,2,0))


def torch2plt_single(img):
    # Assuming img is a PyTorch tensor with shape [1, C, H, W]
    img = img.squeeze(0)  # Remove the batch dimension
    img = img.permute(1, 2, 0)  # Change from [C, H, W] to [H, W, C]
    img = img.numpy()
    return img

################################################################################
################################################################################

def mask2binary(mask, thres):
    # thresholding
    mask = mask.numpy().squeeze()
    outMask = np.zeros(mask.shape)
    outMask[mask>thres] = 1
    return outMask

################################################################################
################################################################################

def plot_predictions(prntNum, dataX, dataY, dataPred, jj, thres):
    '''
    Plot the raw predicitons that allow you to view the activations through the
    layers.
    '''
    # plot the base data
    dataX, dataY, dataPred = dataX[prntNum], dataY[prntNum], dataPred[prntNum]
    fig = plt.figure(figsize=(8,8))
    ax1 = fig.add_subplot(111)
    ax1.imshow(torch2plt(dataX))
    ax1.scatter(mask_to_uv(dataY[0,...])[0],mask_to_uv(dataY[0,...])[1],s=5,color='r',alpha=0.5)
    
    # Binarize the prediction for the specified layer and plot
    predMask = mask2binary(dataPred[jj],thres)
    ax1.scatter(mask_to_uv(predMask)[0],mask_to_uv(predMask)[1],s=5,color='b',alpha=0.05)

    # plot the individual layers
    fig = plt.figure(figsize=(14,8))
    ax1 = fig.add_subplot(111)
    ax1.imshow(torch2plt(tutils.make_grid(dataPred, nrow=3, padding=0)))
    
    # Add text annotations to indicate layer numbers
    # _, xMax = ax1.get_xlim()
    # yMin, _ = ax1.get_ylim()
    # jj = 0
    # ii = 0
    # for prntNum, _ in enumerate(dataPred):
    #     ax1.text(ii*xMax/3 + xMax/50, jj * yMin/2 + yMin/15, prntNum.__str__(),fontdict={'color':'r','size':18,'weight':'bold'})
    #     if ii == 2:
    #         jj += 1
    #         ii = 0
    #     else:
    #         ii += 1

################################################################################
################################################################################

def plot_refined_predictions(prntNum, dataX, dataY, dataPred, thres,cvClean=False,imReturn=False):
    '''
    Plot the refined and final predicitons through a weighted combination and
    thresholding of the layers. (Layers 2 and 3).
    '''
    # plot up the base data
    dataX, dataY, dataPred = dataX[prntNum], dataY[prntNum], dataPred[prntNum]
    fig = plt.figure(figsize=(8,8))
    ax1 = fig.add_subplot(111)
    ax1.axis('off')
    ax1.imshow(torch2plt(dataX))
    #ax1.scatter(mask_to_uv(dataY[0,...])[0],mask_to_uv(dataY[0,...])[1],s=5,color='r',alpha=0.5)

    if cvClean:
        #combined = dataPred[1] * 0.75 + dataPred[2] * 0.25
        #combined = (dataPred[1] * 0.2 + dataPred[2] * 0.3 + dataPred[3] * 0.3 + dataPred[4] * 0.2)
        combined = (dataPred[2] * 0.4 + dataPred[3] * 0.4 + dataPred[4] * 0.1 + dataPred[5] * 0.1)
        
        cvIm = (combined.numpy().squeeze() * 255).astype(np.uint8)

        # Apply Gaussian blur to smooth the image
        cvIm = cv2.GaussianBlur(cvIm, (5, 5), 0)

        # Find the shoreline blobs as contours
        ret, thresh = cv2.threshold(cvIm, int(thres * 255), 255, cv2.THRESH_BINARY)

        # Use morphological operations to refine contours
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # Find the refined contours
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours by geometric properties
        isConvex = np.array([cv2.isContourConvex(_) for _ in contours])
        area = np.array([cv2.contourArea(_) for _ in contours])
        perimeter = np.array([cv2.arcLength(_, True) for _ in contours])
        radius = []
        for _ in contours:
            _, tmpRadius = cv2.minEnclosingCircle(_)
            radius.append(tmpRadius)
        # Filter contours based on radius and perimeter
        contInd = np.where((np.array(radius) > 0.25 * np.array(radius).max()) & (perimeter > 0.25 * perimeter.max()))[0]
        # If no contours meet the criteria, fallback to the largest contour by shape
        if contInd.shape[0] < 1:
            contInd = np.array([np.array([_.shape[0] for _ in contours]).argmax()])
        
        # Binarize the combined predictions using the threshold
        predMask = mask2binary(combined, thres)
        # Extract the coordinates of the positive predictions
        predU = mask_to_uv(predMask)[0]
        predV = mask_to_uv(predMask)[1]
        
        # Boolean array to store whether each prediction point is within a valid contour
        contBool = np.full((predU.shape[0],), False)
        for ii, (thisU, thisV) in enumerate(zip(predU, predV)):
            thisBools = []
            for _ in contInd:
                thisBools.append(cv2.pointPolygonTest(contours[_], (thisU, thisV), True) > 0)
            contBool[ii] = np.any(thisBools)
        
        # Scatter plot the points that are within the valid contours
        ax1.scatter(predU[contBool], predV[contBool], s=5, color='m', alpha=0.07)
    
    else:
        # # perform a weighted combination
        combined = dataPred[1]* 0.5 + dataPred[2] * 0.5
        predMask = mask2binary(combined,thres)
        ax1.scatter(mask_to_uv(predMask)[0],mask_to_uv(predMask)[1],s=5,color='m',alpha=0.07)

    if imReturn:
        # this is for writing a gif output
        ax1.axis('off')
        fig.canvas.draw()
        imData = np.fromstring(fig.canvas.tostring_rgb(), dtype='uint8')
        width, height = fig.canvas.get_width_height()
        midW = int(np.round(width/2))
        midH = int(np.round(height/2))
        imData = imData.reshape((width, height, 3))
        startH = np.where(imData[midW,:,0]<255)[0][0]
        endH = np.where(imData[midW,:,0]<255)[0][-1]
        startW = np.where(imData[:,midH,0]<255)[0][0]
        endW = np.where(imData[:,midH,0]<255)[0][-1]
        imData = imData[startW:endW,startH:endH,:]
        plt.close()
        return imData

def plot_refined_single_prediction(dataX, dataPred, thres,cvClean=False,imReturn=False):
    '''
    Plot the refined and final predicitons through a weighted combination and
    thresholding of the layers. (Layers 2 and 3).
    '''
    # plot up the base data
    dataX = torch2plt_single(dataX)
    # plot up the base data
    fig = plt.figure(figsize=(8,8))
    ax1 = fig.add_subplot(111)
    ax1.axis('off')
    ax1.imshow(dataX)
    #ax1.scatter(mask_to_uv(dataY[0,...])[0],mask_to_uv(dataY[0,...])[1],s=5,color='r',alpha=0.5)

    if cvClean:
        #combined = dataPred[1] * 0.75 + dataPred[2] * 0.25
        #combined = (dataPred[1] * 0.2 + dataPred[2] * 0.3 + dataPred[3] * 0.3 + dataPred[4] * 0.2)
        #combined = (dataPred[2] * 0.4 + dataPred[3] * 0.4 + dataPred[4] * 0.1 + dataPred[5] * 0.1)
        combined = (dataPred[:, 2, :, :] * 0.4 + dataPred[:, 3, :, :] * 0.4 + dataPred[:, 4, :, :] * 0.1 + dataPred[:, 5, :, :] * 0.1)
        
        cvIm = (combined.squeeze(0).numpy() * 255).astype(np.uint8)

        # Apply Gaussian blur to smooth the image
        cvIm = cv2.GaussianBlur(cvIm, (5, 5), 0)

        # Find the shoreline blobs as contours
        ret, thresh = cv2.threshold(cvIm, int(thres * 255), 255, cv2.THRESH_BINARY)

        # Use morphological operations to refine contours
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # Find the refined contours
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours by geometric properties
        isConvex = np.array([cv2.isContourConvex(_) for _ in contours])
        area = np.array([cv2.contourArea(_) for _ in contours])
        perimeter = np.array([cv2.arcLength(_, True) for _ in contours])
        radius = []
        for _ in contours:
            _, tmpRadius = cv2.minEnclosingCircle(_)
            radius.append(tmpRadius)
        # Filter contours based on radius and perimeter
        contInd = np.where((np.array(radius) > 0.25 * np.array(radius).max()) & (perimeter > 0.25 * perimeter.max()))[0]
        # If no contours meet the criteria, fallback to the largest contour by shape
        if contInd.shape[0] < 1:
            contInd = np.array([np.array([_.shape[0] for _ in contours]).argmax()])
        
        # Binarize the combined predictions using the threshold
        predMask = mask2binary(combined.squeeze(0), thres)
        # Extract the coordinates of the positive predictions
        predU = mask_to_uv(predMask)[0]
        predV = mask_to_uv(predMask)[1]
        
        # Boolean array to store whether each prediction point is within a valid contour
        contBool = np.full((predU.shape[0],), False)
        for ii, (thisU, thisV) in enumerate(zip(predU, predV)):
            thisBools = []
            for _ in contInd:
                thisBools.append(cv2.pointPolygonTest(contours[_], (thisU, thisV), True) > 0)
            contBool[ii] = np.any(thisBools)
        
        # Scatter plot the points that are within the valid contours
        ax1.scatter(predU[contBool], predV[contBool], s=5, color='m', alpha=0.07)
    
    else:
        # Perform a weighted combination
        combined = dataPred[:, 1, :, :] * 0.5 + dataPred[:, 2, :, :] * 0.5
        predMask = mask2binary(combined.squeeze(0), thres)
        ax1.scatter(mask_to_uv(predMask)[0], mask_to_uv(predMask)[1], s=5, color='m', alpha=0.07)

    if imReturn:
        # this is for writing a gif output
        ax1.axis('off')
        fig.canvas.draw()
        imData = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        width, height = fig.canvas.get_width_height()
        imData = imData.reshape((height, width, 3))
        imData = cv2.cvtColor(imData, cv2.COLOR_RGB2BGR)
        startH = np.where(imData[int(width / 2), :, 0] < 255)[0][0]
        endH = np.where(imData[int(width / 2), :, 0] < 255)[0][-1]
        startW = np.where(imData[:, int(height / 2), 0] < 255)[0][0]
        endW = np.where(imData[:, int(height / 2), 0] < 255)[0][-1]
        imData = imData[startW:endW, startH:endH, :]
        plt.close()
        return imData

################################################################################
################################################################################

def write_output_gif(gifName,dataX, dataY, dataPred, thres,cvClean=False):
    # write the final prediction into a gif for visualisation
    imageio.mimsave(gifName,
                    [plot_refined_predictions(_,dataX, dataY, dataPred, thres,cvClean,imReturn=True) for _ in np.arange(dataX.shape[0])],
                    fps=0.5)
