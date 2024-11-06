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

def plot_refined_single_prediction(dataX, dataPred, thres,cvClean=False,imReturn=False):
    '''
    Plot the refined and final predicitons through a weighted combination and
    thresholding of the layers. (Layers 2 and 3).
    '''
    # plot up the base data
    dataX = torch2plt_single(dataX)
    dataPred = dataPred[0]
    # plot up the base data
    fig = plt.figure(figsize=(8,8))
    ax1 = fig.add_subplot(111)
    ax1.axis('off')
    ax1.imshow(dataX)

    if cvClean:
        combined = (dataPred[2] * 0.4 + dataPred[3] * 0.4 + dataPred[4] * 0.1 + dataPred[5] * 0.1)
        cvIm = (combined.squeeze(0).numpy() * 255).astype(np.uint8)

        # Apply Gaussian blur to smooth the image
        cvIm = cv2.GaussianBlur(cvIm, (15, 15), 0)
        ret, thresh = cv2.threshold(cvIm, int(thres * 255), 255, cv2.THRESH_BINARY)
        
        # Perform skeletonization to get a thin, consistent line
        skeleton = cv2.ximgproc.thinning(thresh, 0)
        
        skeleton_coords = np.column_stack(np.where(skeleton > 0)) 
        ax1.scatter(skeleton_coords[:, 1], skeleton_coords[:, 0], s=1, color='m', alpha=0.4)

    else:
        # Perform a weighted combination
        combined = (dataPred[2] * 0.4 + dataPred[3] * 0.4 + dataPred[4] * 0.1 + dataPred[5] * 0.1)
        predMask = mask2binary(combined.squeeze(0), thres)
        ax1.scatter(mask_to_uv(predMask)[0], mask_to_uv(predMask)[1], s=7, color='darkmagenta', alpha=0.07)

    if imReturn:
        # this is for writing a gif output
        ax1.axis('off')
        fig.canvas.draw()
        imData = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        width, height = fig.canvas.get_width_height()
        imData = imData.reshape((height*2, width*2, 3))
        imData = cv2.cvtColor(imData, cv2.COLOR_RGB2BGR)
        startH = np.where(imData[int(width / 2), :, 0] < 255)[0][0]
        endH = np.where(imData[int(width / 2), :, 0] < 255)[0][-1]
        startW = np.where(imData[:, int(height / 2), 0] < 255)[0][0]
        endW = np.where(imData[:, int(height / 2), 0] < 255)[0][-1]
        imData = imData[startW:endW, startH:endH, :]
        plt.close()
        return imData
    
def filter_short_contours(skeleton, min_contour_length=50):
    """
    Filters out shorter contours based on a minimum contour length.
    :param skeleton: Skeletonized binary image.
    :param min_contour_length: Minimum length of contours to keep.
    :return: Image with only long contours retained.
    """
    # Find contours in the skeletonized image
    contours, _ = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a blank image to draw the filtered contours
    filtered_skeleton = np.zeros_like(skeleton)
    
    # Filter and draw contours based on length
    for contour in contours:
        if cv2.arcLength(contour, closed=False) > min_contour_length:
            cv2.drawContours(filtered_skeleton, [contour], -1, 255, thickness=1)
    
    return filtered_skeleton

def get_adaptive_threshold(prediction, base_confidence=0.4, percentile=80):
    """
    Calculate an adaptive threshold based on high-confidence scores above a base threshold.
    :param prediction: The predicted confidence map (2D array).
    :param base_confidence: Minimum confidence score to consider a pixel as part of the shoreline.
    :param percentile: Percentile value for adaptive thresholding (e.g., 80 for top 20%).
    :return: Calculated adaptive threshold value.
    """
    # Flatten and filter out low-confidence pixels below the base confidence
    high_conf_pixels = prediction[prediction > base_confidence]

    # Check if there are enough high-confidence pixels to apply the threshold
    if high_conf_pixels.size > 0:
        # Calculate the specified percentile within the high-confidence pixels
        adaptive_thres = np.percentile(high_conf_pixels, percentile)
    else:
        # Fallback in case there are no high-confidence pixels
        adaptive_thres = base_confidence  # or set to a default value like 0.5

    return adaptive_thres


def plot_confidence_distribution(prediction, num_bins=100):
    """
    Plot the cumulative distribution of confidence scores.
    :param prediction: The predicted confidence map (2D array).
    :param num_bins: Number of bins for the histogram.
    """
    # Flatten the prediction to a 1D array
    confidence_scores = prediction.flatten()
    
    # Calculate histogram and cumulative sum
    counts, bin_edges = np.histogram(confidence_scores, bins=num_bins, range=(0, 1))
    cdf = np.cumsum(counts) / np.sum(counts)  # Normalize to get percentages
    
    # Plot the cumulative distribution function (CDF)
    plt.figure(figsize=(10, 6))
    plt.plot(bin_edges[1:], cdf * 100, marker='o', color='b')  # Multiply by 100 for percentage
    plt.xlabel("Confidence Score")
    plt.ylabel("Percentage of Pixels (%)")
    plt.title("Cumulative Distribution of Confidence Scores")
    plt.grid(True)
    plt.show()


def plot_single_prediction(dataX, dataPred, thres, cvClean=False, imReturn=False):
    '''
    Plot the refined and final predictions through a weighted combination and
    thresholding of the layers. (Layers 2 and 3).
    '''
    # Convert dataX for display
    dataX = torch2plt_single(dataX)
    dataPred = dataPred[0]

    # Initialize a figure with subplots for each image
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    # axs[0, 0].imshow(dataX, cmap='gray')
    # axs[0, 0].set_title("Original Frame")
    # axs[0, 0].axis('off')
    
    axs[1, 1].imshow(dataX, cmap='gray')
    axs[1, 1].set_title("After Thinning")
    axs[1, 1].axis('off')
    
    axs[1, 2].imshow(dataX, cmap='gray')
    axs[1, 2].set_title("After Filtering")
    axs[1, 2].axis('off')

    if cvClean:
        # Weighted combination
        combined = (dataPred[2] * 0.4 + dataPred[3] * 0.4 + dataPred[4] * 0.1 + dataPred[5] * 0.1)
        
        adaptive_thres_value = get_adaptive_threshold(combined.squeeze(0).cpu().numpy(), base_confidence=0.4, percentile=50)
        print("adaptive_thres_value:", adaptive_thres_value)
        #plot_confidence_distribution(combined.squeeze(0).cpu().numpy())
        
        heatmap = axs[0, 0].imshow(combined.squeeze(0).cpu().numpy(), cmap='hot', interpolation='nearest')
        fig.colorbar(heatmap, ax=axs[0, 0], label="Confidence Score")
        axs[0, 0].set_title("Confidence Heatmap")
        axs[0, 0].axis('off')
        
        # Display the combined prediction
        axs[0, 1].imshow(combined.squeeze(0).cpu().numpy(), cmap='gray')
        axs[0, 1].set_title("Combined Prediction")
        axs[0, 1].axis('off')
        
        # Convert to cvIm for OpenCV operations
        cvIm = (combined.squeeze(0).numpy() * 255).astype(np.uint8)

        # Apply Gaussian blur
        cvIm_blurred = cv2.GaussianBlur(cvIm, (7, 7), 0)
        
        # Display cvIm after Gaussian blur
        # axs[0, 2].imshow(cvIm_blurred, cmap='gray')
        # axs[0, 2].set_title("After Gaussian Blur")
        # axs[0, 2].axis('off')
        
        thresh_new = cv2.adaptiveThreshold(cvIm_blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
        cv2.THRESH_BINARY,21,2)
        axs[0, 2].imshow(thresh_new, cmap='gray')
        axs[0, 2].set_title("After Adaptive Thresh")
        axs[0, 2].axis('off')

        # Apply threshold
        ret, thresh = cv2.threshold(cvIm_blurred, int(adaptive_thres_value * 255), 255, cv2.THRESH_BINARY)
        
        # Display thresholded image
        axs[1, 0].imshow(thresh, cmap='gray')
        axs[1, 0].set_title("Thresholded Image")
        axs[1, 0].axis('off')
        
        # Perform skeletonization for thin contours
        skeleton = cv2.ximgproc.thinning(thresh, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
        skeleton_ori = np.column_stack(np.where(skeleton > 0)) 
        axs[1, 1].scatter(skeleton_ori[:, 1], skeleton_ori[:, 0], s=1, color='m', alpha=0.4)

        skeleton = filter_short_contours(skeleton, min_contour_length=200)
        # Scatter plot for skeleton coordinates (if needed)
        skeleton_coords = np.column_stack(np.where(skeleton > 0)) 
        axs[1, 2].scatter(skeleton_coords[:, 1], skeleton_coords[:, 0], s=1, color='m', alpha=0.4)

    else:
        # Perform a weighted combination and plot as a scatter if cvClean is False
        combined = (dataPred[2] * 0.4 + dataPred[3] * 0.4 + dataPred[4] * 0.1 + dataPred[5] * 0.1)
        predMask = mask2binary(combined.squeeze(0), thres)
        axs[3].scatter(mask_to_uv(predMask)[0], mask_to_uv(predMask)[1], s=7, color='darkmagenta', alpha=0.07)

    plt.tight_layout()
    plt.show()

    if imReturn:
        fig.canvas.draw()
        imData = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        width, height = fig.canvas.get_width_height()
        imData = imData.reshape((height, width, 3))
        imData = cv2.cvtColor(imData, cv2.COLOR_RGB2BGR)
        plt.close()
        return imData


################################################################################
################################################################################

def write_output_gif(gifName,dataX, dataY, dataPred, thres,cvClean=False):
    # write the final prediction into a gif for visualisation
    imageio.mimsave(gifName,
                    [plot_refined_predictions(_,dataX, dataY, dataPred, thres,cvClean,imReturn=True) for _ in np.arange(dataX.shape[0])],
                    fps=0.5)

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
        # perform a weighted combination
        combined = (dataPred[2] * 0.4 + dataPred[3] * 0.4 + dataPred[4] * 0.1 + dataPred[5] * 0.1)
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