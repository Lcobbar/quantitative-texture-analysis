import numpy as np
import SimpleITK as sitk
from skimage import feature
from radiomics import imageoperations
from scipy import ndimage as ndi
from scipy.ndimage import binary_erosion


def pre_process(image, mask, yaml, normalize=False):
    """
    Normalize all image with z-score normalization with removing outliers and interpolates

    :param normalize: activates normalization
    :param image: SimpleITK image
    :param mask: SimpleITK image mask
    :param yaml: settings file of PyRadiomics
    :return: roi interpolated, image interpolated, mask interpolated
    """

    if normalize:
        image = imageoperations.normalizeImage(image, **{'normalizeScale': yaml['setting']['normalizeScale']})

    (ii, im) = imageoperations.resampleImage(image, mask, **{"resampledPixelSpacing": yaml['setting']['resampledPixelSpacing'], "label": 1,
                                                             'interpolator': yaml['setting']['interpolator']})

    array_ii = sitk.GetArrayFromImage(ii[:, :, 0])
    array_im = sitk.GetArrayFromImage(im[:, :, 0])

    masked_image = array_ii * array_im

    return masked_image, array_ii, array_im


def erode_mask(image, pixels=5):
    """
      Erode image
      :param image: 2D image to erode
      :param pixels: number of pixels. Default 5
      :return: 2D eroded image
      """

    erosion_footprint = ndi.generate_binary_structure(image.ndim, image.ndim)
    eroded_image = image.copy()
    eroded_image = binary_erosion(eroded_image, erosion_footprint, iterations=pixels)

    return eroded_image * 1


# https://pyimagesearch.com/2015/12/07/local-binary-patterns-with-python-opencv/
def lbp_features_extractor(image, mask):
    """
    Compute the uniform LBP histogram features (R=1, P=8). The mask is eroded to avoid the border effect.
    :param image: 2D array image
    :param mask: 2D array image with 1 in ROI pixels and 0 in non-ROI pixels
    :return: features dictionary (Energy,Entropy and LBP normalized histogram)
    """

    features = {}
    R = [1, 2, 3]
    P = [8, 16, 24]

    for k in range(3):
        erosion_footprint = ndi.generate_binary_structure(mask.ndim, mask.ndim)
        eroded_mask = mask.copy()
        img = image.copy()

        eroded_mask = binary_erosion(eroded_mask, erosion_footprint, border_value=0)

        lbp = feature.local_binary_pattern(img, P[k], R[k], 'uniform')
        lbp_masked = lbp * eroded_mask
        lbp_roi1d = lbp_masked[eroded_mask == np.max(eroded_mask)]

        # Values in LBP images ranges from [0, numPoints + 2], a value for each of the possible numPoints + 1
        # (possible rotation invariant prototypes along with an extra dimension for all patterns that are not uniform)
        hist, _ = np.histogram(lbp_roi1d, bins=range(0, P[k] + 3), range=(0, P[k] + 2))
        eps = 1e-7
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)

        energy_label = f'LBP_R_{R[k]}_P_{P[k]}_Energy'
        entropy_label = f'LBP_R_{R[k]}_P_{P[k]}_Entropy'
        hist_label = f'LBP_R_{R[k]}_P_{P[k]}_Hist'
        features[energy_label] = np.sum(hist ** 2)
        features[entropy_label] = -np.sum(hist * np.log2(hist + 1e-9))

        for i, v in enumerate(hist, start=1):
            key = f'{hist_label}_{i}'
            features[key] = v

    return features