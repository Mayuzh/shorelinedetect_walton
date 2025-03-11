import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as tutils

import imageio

import cv2
import torch
import time
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

def plot_refined_single_prediction(dataX, dataPred, thres, cvClean=False, imReturn=False):
    '''
    Plot the refined and final predicitons through a weighted combination and
    thresholding of the layers. (Layers 2 and 3).
    '''
    start_time = time.time()
    # Convert PyTorch tensor to NumPy for processing
    dataX = torch2plt_single(dataX)
    dataPred = dataPred[0]
    # plot up the base data
    fig = plt.figure(figsize=(8,8))
    ax1 = fig.add_subplot(111)
    ax1.axis('off')
    ax1.imshow(dataX)

    if cvClean:
        combined = (dataPred[2] * 0.4 + dataPred[3] * 0.4 + dataPred[4] * 0.1 + dataPred[5] * 0.1)
        combined_array = combined.squeeze(0).cpu().numpy()

        # Apply Gaussian blur to smooth the image
        cvIm = (combined_array * 255).astype(np.uint8)
        cvIm = cv2.GaussianBlur(cvIm, (13, 13), 0)

        # Adaptive thresholding
        adaptive_thres_value = get_adaptive_threshold(combined.squeeze(0).cpu().numpy(), base_confidence=0.3, percentile=60)
        _, thresh = cv2.threshold(cvIm, int(adaptive_thres_value * 255), 255, cv2.THRESH_BINARY)
        #_, thresh = cv2.threshold(cvIm, int(0.8 * 255), 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        
        # Perform skeletonization to get a thin, consistent line
        skeleton = cv2.ximgproc.thinning(thresh, 0)
        skeleton = filter_contours(skeleton) 
        #skeleton = filter_short_contours(skeleton, min_contour_length=400) 
        
        # Scatter plot for visualization
        skeleton_coords = np.column_stack(np.where(skeleton > 0)) 
        ax1.scatter(skeleton_coords[:, 1], skeleton_coords[:, 0], s=1, color='m', alpha=0.5)
    else:
        # Perform a weighted combination
        combined = (dataPred[2] * 0.4 + dataPred[3] * 0.4 + dataPred[4] * 0.1 + dataPred[5] * 0.1)
        predMask = mask2binary(combined.squeeze(0), thres)
        shoreline_coords = mask_to_uv(predMask)  # (x, y) format
        ax1.scatter(shoreline_coords[0], shoreline_coords[1], s=1, color='darkmagenta', alpha=0.07)

    if imReturn:
        # Generate image for output
        ax1.axis('off')
        fig.canvas.draw()
        imData = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')  
        width, height = fig.canvas.get_width_height()
        imData = imData.reshape((height, width, 3))
        imData = cv2.cvtColor(imData, cv2.COLOR_RGB2BGR)
        plt.close()

        # Cropping based on the non-white region
        startH = np.where(imData[int(width / 2), :, 0] < 255)[0][0]
        endH = np.where(imData[int(width / 2), :, 0] < 255)[0][-1]
        startW = np.where(imData[:, int(height / 2), 0] < 255)[0][0]
        endW = np.where(imData[:, int(height / 2), 0] < 255)[0][-1]
        imData = imData[startW:endW, startH:endH, :]

        # Adjust shoreline coordinates to match the cropped image
        if cvClean:
            adjusted_coords = skeleton_coords - np.array([startW, startH])  # (y, x) format
            adjusted_coords = adjusted_coords[:, [1, 0]]  # Convert to (x, y) format
        else:
            adjusted_coords = shoreline_coords - np.array([startH, startW])  # (x, y) format

        #post_time = time.time() - start_time
        #print(f"post_time: {post_time:.4f} seconds")
                             
        #return imData       
        return imData, adjusted_coords
                                                                                                
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


def filter_contours(skeleton, angle_threshold=20, proximity_threshold=30):
    """
    Filters out contours to construct the longest, smooth, connected shoreline by keeping parallel segments
    and merging the closest ones, while removing all unrelated contours.
    :param skeleton: Skeletonized binary image.
    :param angle_threshold: Maximum allowable angle (in degrees) for smooth alignment.
    :param proximity_threshold: Maximum allowable distance between segments for merging.
    :return: Image with the constructed longest shoreline.
    """
    # Find all contours in the skeletonized image
    contours, _ = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Identify the longest contour as the primary shoreline
    max_contour = max(contours, key=lambda cnt: cv2.arcLength(cnt, closed=False))

    # Function to calculate angle between two vectors
    def calculate_angle(v1, v2):
        dot_product = np.dot(v1, v2)
        norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm_product == 0:
            return 180  # Treat zero-length vectors as perpendicular
        cos_theta = np.clip(dot_product / norm_product, -1.0, 1.0)
        return np.degrees(np.arccos(cos_theta))

    # Helper function to calculate Euclidean distance
    def euclidean_dist(point1, point2):
        return np.linalg.norm(np.array(point1) - np.array(point2))

    # Create a blank image to draw the filtered contours
    filtered_skeleton = np.zeros_like(skeleton)

    # Keep track of merged segments
    merged_contours = [max_contour]

    # Primary vector of the longest contour
    primary_start = max_contour[0][0]
    primary_end = max_contour[-1][0]
    primary_vector = np.array(primary_end) - np.array(primary_start)

    # Identify parallel contours
    parallel_contours = []
    for contour in contours:
        if np.array_equal(contour, max_contour):
            continue  # Skip the primary shoreline

        # Get the start and end points of the current contour
        contour_start = contour[0][0]
        contour_end = contour[-1][0]
        contour_vector = np.array(contour_end) - np.array(contour_start)

        # Check if the contour is parallel to the primary shoreline
        angle = calculate_angle(primary_vector, contour_vector)
        if angle < angle_threshold:
            parallel_contours.append(contour)

    # Merge parallel contours based on proximity
    while parallel_contours:
        closest_contour = None
        closest_distance = float('inf')

        for contour in parallel_contours:
            # Measure proximity to the current merged contours
            for merged in merged_contours:
                distance_start = euclidean_dist(contour[0][0], merged[-1][0])
                distance_end = euclidean_dist(contour[-1][0], merged[-1][0])

                if closest_contour is not None and closest_distance < proximity_threshold:
                    closest_distance = min(distance_start, distance_end)
                    closest_contour = contour

        # Merge the closest contour if within proximity threshold
        if closest_contour and closest_distance < proximity_threshold:
            merged_contours.append(closest_contour)
            parallel_contours.remove(closest_contour)
        else:
            break  # Stop merging if no contour is close enough

    # Draw the final merged shoreline
    for contour in merged_contours:
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

def plot_single_prediction(dataX, dataPred, thres, cvClean=False, imReturn=False):
    '''
    Plot the refined and final predictions through a weighted combination and
    thresholding of the layers. (Layers 2 and 3).
    '''
    # Convert dataX for display
    dataX = torch2plt_single(dataX)
    dataPred = dataPred[0]

    # Initialize a figure with subplots for each image
    fig, axs = plt.subplots(4, 4, figsize=(24, 10))
    axs[3, 0].imshow(dataX, cmap='gray')
    axs[3, 0].set_title("After Thinning")
    axs[3, 0].axis('off')
    axs[3, 1].imshow(dataX, cmap='gray')
    axs[3, 1].set_title("After Thinning")
    axs[3, 1].axis('off')
    axs[3, 2].imshow(dataX, cmap='gray')
    axs[3, 2].set_title("After Thinning")
    axs[3, 2].axis('off')
    axs[3, 3].imshow(dataX, cmap='gray')
    axs[3, 3].set_title("After Thinning")
    axs[3, 3].axis('off')

    
    # axs[1, 1].imshow(dataX, cmap='gray')
    # axs[1, 1].set_title("Filtering base on Linearity")
    # axs[1, 1].axis('off')
    
    # axs[1, 2].imshow(dataX, cmap='gray')
    # axs[1, 2].set_title("Filtering base on Length")
    # axs[1, 2].axis('off')

    if cvClean:
        # Weighted combination
        combined = (dataPred[2] * 0.4 + dataPred[3] * 0.4 + dataPred[4] * 0.1 + dataPred[5] * 0.1)
        
        heatmap = axs[0, 0].imshow(combined.squeeze(0).cpu().numpy(), cmap='hot', interpolation='nearest')
        fig.colorbar(heatmap, ax=axs[0, 0], label="Confidence Score")
        axs[0, 0].set_title("Confidence Heatmap")
        axs[0, 0].axis('off')
        
        # Display the combined prediction
        # axs[0, 1].imshow(combined.squeeze(0).cpu().numpy(), cmap='gray')
        # axs[0, 1].set_title("Combined Prediction")
        # axs[0, 1].axis('off')
        
        # Convert to cvIm for OpenCV operations
        cvIm = (combined.squeeze(0).numpy() * 255).astype(np.uint8)
        # Apply Gaussian blur
        cvIm_blurred1 = cv2.GaussianBlur(cvIm, (5, 5), 0)
        cvIm_blurred2 = cv2.GaussianBlur(cvIm, (11, 11), 0)
        cvIm_blurred3 = cv2.GaussianBlur(cvIm, (17, 17), 0)
        cvIm_blurred4 = cv2.GaussianBlur(cvIm, (21, 21), 0)
        
        # Display cvIm after Gaussian blur
        axs[1, 0].imshow(cvIm_blurred1, cmap='gray')
        axs[1, 0].set_title("After Gaussian Blur")
        axs[1, 0].axis('off')
        axs[1, 0].imshow(cvIm_blurred2, cmap='gray')
        axs[1, 1].set_title("After Gaussian Blur")
        axs[1, 1].axis('off')
        axs[1, 1].imshow(cvIm_blurred3, cmap='gray')
        axs[1, 1].set_title("After Gaussian Blur")
        axs[1, 2].axis('off')
        axs[1, 2].imshow(cvIm_blurred4, cmap='gray')
        axs[1, 2].set_title("After Gaussian Blur")
        axs[1, 2].axis('off')
        axs[1, 3].imshow(cvIm_blurred4, cmap='gray')
        axs[1, 3].set_title("After Gaussian Blur")
        axs[1, 3].axis('off')

        adaptive_thres_value = get_adaptive_threshold(combined.squeeze(0).cpu().numpy(), base_confidence=0.4, percentile=50)
        print("adaptive_thres_value:", adaptive_thres_value)
        
        # Apply threshold
        _, thresh1 = cv2.threshold(cvIm_blurred1, int(adaptive_thres_value * 255), 255, cv2.THRESH_BINARY)
        _, thresh_otsu = cv2.threshold(cvIm_blurred1, int(adaptive_thres_value * 255), 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, thresh2 = cv2.threshold(cvIm_blurred2, int(adaptive_thres_value * 255), 255, cv2.THRESH_BINARY)
        _, thresh_otsu = cv2.threshold(cvIm_blurred1, int(adaptive_thres_value * 255), 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, thresh3 = cv2.threshold(cvIm_blurred3, int(adaptive_thres_value * 255), 255, cv2.THRESH_BINARY)
        _, thresh_otsu = cv2.threshold(cvIm_blurred1, int(adaptive_thres_value * 255), 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, thresh4 = cv2.threshold(cvIm_blurred4, int(adaptive_thres_value * 255), 255, cv2.THRESH_BINARY)
        _, thresh_otsu = cv2.threshold(cvIm_blurred1, int(adaptive_thres_value * 255), 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Display thresholded image
        axs[2, 0].imshow(thresh1, cmap='gray')
        axs[2, 0].set_title("Thresholded Image")
        axs[2, 0].axis('off')
        axs[2, 1].imshow(thresh2, cmap='gray')
        axs[2, 1].set_title("Thresholded Image")
        axs[2, 1].axis('off')
        axs[2, 2].imshow(thresh3, cmap='gray')
        axs[2, 2].set_title("Thresholded Image")
        axs[2, 2].axis('off')
        axs[2, 3].imshow(thresh4, cmap='gray')
        axs[2, 3].set_title("Thresholded Image")
        axs[2, 3].axis('off')
        # Display thresholded image
        # axs[2, 1].imshow(thresh_otsu, cmap='gray')
        # axs[2, 1].set_title("Thresholded Otsu Image")
        # axs[2, 1].axis('off')
        
        # Perform skeletonization for thin contours
        skeleton = cv2.ximgproc.thinning(thresh1, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
        skeleton_ori = np.column_stack(np.where(skeleton > 0)) 
        axs[3, 0].scatter(skeleton_ori[:, 1], skeleton_ori[:, 0], s=1, color='m', alpha=0.4)
        skeleton = cv2.ximgproc.thinning(thresh2, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
        skeleton_ori = np.column_stack(np.where(skeleton > 0)) 
        axs[3, 1].scatter(skeleton_ori[:, 1], skeleton_ori[:, 0], s=1, color='m', alpha=0.4)
        skeleton = cv2.ximgproc.thinning(thresh3, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
        skeleton_ori = np.column_stack(np.where(skeleton > 0)) 
        axs[3, 2].scatter(skeleton_ori[:, 1], skeleton_ori[:, 0], s=1, color='m', alpha=0.4)
        skeleton = cv2.ximgproc.thinning(thresh4, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
        skeleton_ori = np.column_stack(np.where(skeleton > 0)) 
        axs[3, 3].scatter(skeleton_ori[:, 1], skeleton_ori[:, 0], s=1, color='m', alpha=0.4)

        # skeleton1 = filter_contours(skeleton)
        # skeleton_coords1 = np.column_stack(np.where(skeleton1 > 0)) 
        # axs[1, 1].scatter(skeleton_coords1[:, 1], skeleton_coords1[:, 0], s=1, color='m', alpha=0.4)
        
        # skeleton2 = filter_short_contours(skeleton, min_contour_length=100) 
        # skeleton_coords2 = np.column_stack(np.where(skeleton2 > 0)) 
        # axs[1, 2].scatter(skeleton_coords2[:, 1], skeleton_coords2[:, 0], s=1, color='m', alpha=0.4)

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