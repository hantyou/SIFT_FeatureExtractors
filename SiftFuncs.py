from math import log, sqrt

from numpy import array, arctan2, exp, dot, log, logical_and, roll, sqrt, stack, trace, rad2deg, \
    where, zeros, round, float32, all
from numpy.linalg import det, lstsq

from ImageFuncs import *
from ShowFuncs import *
import pysift

float_tolerance = 1e-7

"""计算金字塔层数"""


def CalcPyrNum(shape):
    """Compute number of octaves in image pyramid as function of base image shape (OpenCV default)
    """
    return int(round(log(min(shape)) / log(2) - 3))


"""生成金字塔每一层(Octave)不同尺寸(Interval)高斯模糊的sigma值"""


def genGaussianKernelSigmas(sigma, num_intervals):
    """Generate list of gaussian kernels at which to blur the input image. Default values of sigma, intervals, and octaves follow section 3 of Lowe's paper.
    """
    num_images_per_octave = num_intervals + 3
    k = 2 ** (1. / num_intervals)
    gaussian_kernels = np.zeros(
        num_images_per_octave)  # scale of gaussian blur necessary to go from one blur scale to the next within an octave
    gaussian_kernels[0] = sigma

    for image_index in range(1, num_images_per_octave):
        sigma_previous = (k ** (image_index - 1)) * sigma
        sigma_total = k * sigma_previous
        gaussian_kernels[image_index] = sqrt(sigma_total ** 2 - sigma_previous ** 2)
    return gaussian_kernels


"""生成图像金字塔,该函数废弃"""


def GeneratePyrPics(I, LayerNum):
    pyrPics = []
    I0 = cv2.pyrUp(I)
    pyrPics.append(I0)
    pyrPics.append(I)
    for i in range(2, LayerNum):
        pyrPics.append(MyBiLiResize(pyrPics[i - 1], 2))
        # pyrPics.append(cv2.pyrDown(pyrPics[i - 1]))
    return pyrPics


"""生成高斯金字塔,每一层内含不同尺寸模糊的高斯模糊图像"""


def GenerateGausPyrPics(I, NumOctaves, scale=1.5, s=3, gaussKernelSize=3, sigma=1.414213):
    pyrPics = []
    NumOctaves += 1
    IntervalNumPerLayers = s + 3
    GaussSigma = genGaussianKernelSigmas(sigma, s)
    for i in range(NumOctaves):
        PicIntervals = [] # first image in octave already has the correct blur
        for gaussian_kernel in GaussSigma[0:]:
            I = cv2.GaussianBlur(I, (0, 0), sigmaX=gaussian_kernel,
                                 sigmaY=gaussian_kernel)
            PicIntervals.append(I)
        pyrPics.append(PicIntervals)
        octave_base = PicIntervals[-3]
        if i == 0:
            I = cv2.resize(octave_base, (int(octave_base.shape[1] / 2), int(octave_base.shape[0] / 2)),
                           interpolation=cv2.INTER_NEAREST)
        else:
            I = cv2.resize(octave_base, (int(octave_base.shape[1] / scale), int(octave_base.shape[0] / scale)),
                           interpolation=cv2.INTER_NEAREST)
    return np.array(pyrPics)


"""根据生成的高斯金字塔,生成高斯差金字塔"""


def GenerateDoGImages(pyrPics):
    dog_images = []
    for gaussian_images_in_octave in pyrPics:
        dog_images_in_octave = []
        for first_image, second_image in zip(gaussian_images_in_octave, gaussian_images_in_octave[1:]):
            dog_images_in_octave.append(cv2.subtract(second_image,
                                                     first_image))  # ordinary subtraction will not work because the images are unsigned integers
        dog_images.append(dog_images_in_octave)
    return np.array(dog_images)


"""找到每一层的极大值点,并且对每个极值点的邻域,根据二次拟合计算精确的极值点"""


def FindMaxMin(Gauss, DoGs, s, sigma, ImBoarderWidth=5, ContrastThreshold=0.04):
    print("开始寻找极大值")
    threshold = int(0.5 * ContrastThreshold / s * 255)
    # threshold = 0.5 * ContrastThreshold / s
    KeyPoints = []
    for o, CurrentOctave in enumerate(DoGs):
        for i, [layer0, layer1, layer2] in enumerate(zip(CurrentOctave, CurrentOctave[1:], CurrentOctave[2:])):
            # [h, w] = [layer0.shape[0], layer0.shape[1]]
            for y in range(ImBoarderWidth, layer0.shape[0] - ImBoarderWidth):
                for x in range(ImBoarderWidth, layer0.shape[1] - ImBoarderWidth):
                    if IsMaxOrMin(layer0[y - 1:y + 2, x - 1:x + 2], layer1[y - 1:y + 2, x - 1:x + 2],
                                  layer2[y - 1:y + 2, x - 1:x + 2], threshold):
                        # localizeExtremumViaQuadraticFit() 函数计算过程中就完成了
                        # 1. 剔除对比度过小的点;
                        # 2. 剔除边缘曲率过大的点
                        # 这两个步骤。
                        localization_result = localizeExtremumViaQuadraticFit(y, x, i + 1, o, s, CurrentOctave, sigma,
                                                                              ContrastThreshold, ImBoarderWidth)
                        if localization_result is not None:
                            keypoint, localized_image_index = localization_result
                            keypoints_with_orientations = computeKeypointsWithOrientations(
                                keypoint, o, Gauss[o][localized_image_index])
                            for keypoint_with_orientation in keypoints_with_orientations:
                                KeyPoints.append(keypoint_with_orientation)
    return KeyPoints


def IsMaxOrMin(first_subimage, second_subimage, third_subimage, threshold):
    """Return True if the center element of the 3x3x3 input array is strictly greater than or less than all its neighbors, False otherwise
    """
    center_pixel_value = second_subimage[1, 1]
    if abs(center_pixel_value) > threshold:
        if center_pixel_value > 0:
            return all(center_pixel_value >= first_subimage) and \
                   all(center_pixel_value >= third_subimage) and \
                   all(center_pixel_value >= second_subimage[0, :]) and \
                   all(center_pixel_value >= second_subimage[2, :]) and \
                   center_pixel_value >= second_subimage[1, 0] and \
                   center_pixel_value >= second_subimage[1, 2]
        elif center_pixel_value < 0:
            return all(center_pixel_value <= first_subimage) and \
                   all(center_pixel_value <= third_subimage) and \
                   all(center_pixel_value <= second_subimage[0, :]) and \
                   all(center_pixel_value <= second_subimage[2, :]) and \
                   center_pixel_value <= second_subimage[1, 0] and \
                   center_pixel_value <= second_subimage[1, 2]
    return False


r"""
    if center_pixel_value > 0:
        return all(center_pixel_value >= first_subimage) and \
               all(center_pixel_value >= third_subimage) and \
               all(center_pixel_value >= second_subimage[0, :]) and \
               all(center_pixel_value >= second_subimage[2, :]) and \
               center_pixel_value >= second_subimage[1, 0] and \
               center_pixel_value >= second_subimage[1, 2]
    elif center_pixel_value < 0:
        return all(center_pixel_value <= first_subimage) and \
               all(center_pixel_value <= third_subimage) and \
               all(center_pixel_value <= second_subimage[0, :]) and \
               all(center_pixel_value <= second_subimage[2, :]) and \
               center_pixel_value <= second_subimage[1, 0] and \
               center_pixel_value <= second_subimage[1, 2]
"""


def localizeExtremumViaQuadraticFit(y, x, i, o, num_intervals, DoGs, sigma,
                                    ContrastThreshold, ImBoarderWidth, eigenvalue_ratio=10,
                                    num_attempts_until_convergence=5):
    """Iteratively refine pixel positions of scale-space extrema via quadratic fit around each extremum's neighbors
    """
    extremum_is_outside_image = False
    image_shape = DoGs[0].shape
    for attempt_index in range(num_attempts_until_convergence):
        # need to convert from uint8 to float32 to compute derivatives and need to rescale pixel values to [0, 1] to apply Lowe's thresholds
        first_image, second_image, third_image = DoGs[i - 1:i + 2]
        pixel_cube = stack([first_image[y - 1:y + 2, x - 1:x + 2],
                            second_image[y - 1:y + 2, x - 1:x + 2],
                            third_image[y - 1:y + 2, x - 1:x + 2]]).astype('float32') / 255.
        gradient = computeGradientAtCenterPixel(pixel_cube)  # 计算梯度值
        hessian = computeHessianAtCenterPixel(pixel_cube)  # 计算海瑟矩阵
        extremum_update = -lstsq(hessian, gradient, rcond=None)[0]
        if abs(extremum_update[0]) < 0.5 and abs(extremum_update[1]) < 0.5 and abs(extremum_update[2]) < 0.5:
            # 都小于0.5说明没办法继续更新了，所以直接剔除，终止迭代
            break
        x += int(round(extremum_update[0]))
        y += int(round(extremum_update[1]))
        i += int(round(extremum_update[2]))
        # make sure the new pixel_cube will lie entirely within the image
        if y < ImBoarderWidth or y >= image_shape[0] - ImBoarderWidth or x < ImBoarderWidth or x >= \
                image_shape[1] - ImBoarderWidth or i < 1 or i > num_intervals:
            extremum_is_outside_image = True
            break
    if extremum_is_outside_image:
        return None
    if attempt_index >= num_attempts_until_convergence - 1:
        return None
    functionValueAtUpdatedExtremum = pixel_cube[1, 1, 1] + 0.5 * dot(gradient, extremum_update)  # 求出极大点处的值
    if abs(functionValueAtUpdatedExtremum) * num_intervals >= ContrastThreshold:
        xy_hessian = hessian[:2, :2]
        xy_hessian_trace = trace(xy_hessian)
        xy_hessian_det = det(xy_hessian)
        """
        eigenvalue_ratio是一个自定义的阈值,在论文中写作r
        论文中的公式如下:
            Tr(H)^2 * r < Det(H) * (r+1)^2
        H 是取Hessian矩阵的左上角的二阶主子式,也就是去除s分量的影响,只保留x,y
        """
        if xy_hessian_det > 0 and eigenvalue_ratio * (xy_hessian_trace ** 2) < (
                (eigenvalue_ratio + 1) ** 2) * xy_hessian_det:
            # Contrast check passed -- construct and return OpenCV KeyPoint object
            keypoint = cv2.KeyPoint()
            keypoint.pt = (
                (x + extremum_update[0]) * (2 ** o), (y + extremum_update[1]) * (2 ** o))
            # 这里乘了2的o次方,表示归到金字塔底层图像
            # keypoint.octave = o + i * (2 ** 8) + int(round((extremum_update[2] + 0.5) * 255)) * (2 ** 16)
            keypoint.octave = o
            keypoint.size = sigma * (2 ** ((i + extremum_update[2]) / float32(num_intervals))) * (
                    2 ** (o + 1))  # octave_index + 1 because the input image was doubled
            keypoint.response = abs(functionValueAtUpdatedExtremum)
            return keypoint, i
    return None


def computeGradientAtCenterPixel(pixel_array):
    """Approximate gradient at center pixel [1, 1, 1] of 3x3x3 array using central difference formula of order O(h^2), where h is the step size
    """
    # With step size h, the central difference formula of order O(h^2) for f'(x) is (f(x + h) - f(x - h)) / (2 * h)
    # Here h = 1, so the formula simplifies to f'(x) = (f(x + 1) - f(x - 1)) / 2
    # NOTE: x corresponds to second array axis, y corresponds to first array axis, and s (scale) corresponds to third array axis
    dx = 0.5 * (pixel_array[1, 1, 2] - pixel_array[1, 1, 0])
    dy = 0.5 * (pixel_array[1, 2, 1] - pixel_array[1, 0, 1])
    ds = 0.5 * (pixel_array[2, 1, 1] - pixel_array[0, 1, 1])
    return array([dx, dy, ds])


def computeHessianAtCenterPixel(pixel_array):
    """Approximate Hessian at center pixel [1, 1, 1] of 3x3x3 array using central difference formula of order O(h^2), where h is the step size
    """
    # With step size h, the central difference formula of order O(h^2) for f''(x) is (f(x + h) - 2 * f(x) + f(x - h)) / (h ^ 2)
    # Here h = 1, so the formula simplifies to f''(x) = f(x + 1) - 2 * f(x) + f(x - 1)
    # With step size h, the central difference formula of order O(h^2) for (d^2) f(x, y) / (dx dy) = (f(x + h, y + h) - f(x + h, y - h) - f(x - h, y + h) + f(x - h, y - h)) / (4 * h ^ 2)
    # Here h = 1, so the formula simplifies to (d^2) f(x, y) / (dx dy) = (f(x + 1, y + 1) - f(x + 1, y - 1) - f(x - 1, y + 1) + f(x - 1, y - 1)) / 4
    # NOTE: x corresponds to second array axis, y corresponds to first array axis, and s (scale) corresponds to third array axis
    center_pixel_value = pixel_array[1, 1, 1]
    dxx = pixel_array[1, 1, 2] - 2 * center_pixel_value + pixel_array[1, 1, 0]
    dyy = pixel_array[1, 2, 1] - 2 * center_pixel_value + pixel_array[1, 0, 1]
    dss = pixel_array[2, 1, 1] - 2 * center_pixel_value + pixel_array[0, 1, 1]
    dxy = 0.25 * (pixel_array[1, 2, 2] - pixel_array[1, 2, 0] - pixel_array[1, 0, 2] + pixel_array[1, 0, 0])
    dxs = 0.25 * (pixel_array[2, 1, 2] - pixel_array[2, 1, 0] - pixel_array[0, 1, 2] + pixel_array[0, 1, 0])
    dys = 0.25 * (pixel_array[2, 2, 1] - pixel_array[2, 0, 1] - pixel_array[0, 2, 1] + pixel_array[0, 0, 1])
    return array([[dxx, dxy, dxs],
                  [dxy, dyy, dys],
                  [dxs, dys, dss]])


def computeKeypointsWithOrientations(keypoint, octave_index, gaussian_image, radius_factor=3, num_bins=36,
                                     peak_ratio=0.8, scale_factor=1.5):
    """Compute orientations for each keypoint
    """
    """
    疑问:
    1. scale_factor有什么含义,为什么是1.5,之前金字塔的放缩尺度不是2嘛
    2. 
    """
    keypoints_with_orientations = []
    image_shape = gaussian_image.shape

    scale = scale_factor * keypoint.size / float32(
        2 ** (octave_index + 1))  # compare with keypoint.size computation in localizeExtremumViaQuadraticFit()
    radius = int(round(radius_factor * scale))
    weight_factor = -0.5 / (scale ** 2)
    raw_histogram = zeros(num_bins)
    smooth_histogram = zeros(num_bins)

    for i in range(-radius, radius + 1):
        region_y = int(round(keypoint.pt[1] / float32(2 ** octave_index))) + i
        if region_y > 0 and region_y < image_shape[0] - 1:
            for j in range(-radius, radius + 1):
                region_x = int(round(keypoint.pt[0] / float32(2 ** octave_index))) + j
                if region_x > 0 and region_x < image_shape[1] - 1:
                    dx = gaussian_image[region_y, region_x + 1] - gaussian_image[region_y, region_x - 1]
                    dy = gaussian_image[region_y - 1, region_x] - gaussian_image[region_y + 1, region_x]
                    gradient_magnitude = sqrt(dx * dx + dy * dy)
                    gradient_orientation = rad2deg(arctan2(dy, dx))
                    weight = exp(weight_factor * (
                            i ** 2 + j ** 2))  # constant in front of exponential can be dropped because we will find peaks later
                    histogram_index = int(round(gradient_orientation * num_bins / 360.))
                    raw_histogram[histogram_index % num_bins] += weight * gradient_magnitude

    for n in range(num_bins):
        smooth_histogram[n] = (6 * raw_histogram[n] + 4 * (raw_histogram[n - 1] + raw_histogram[(n + 1) % num_bins]) +
                               raw_histogram[n - 2] + raw_histogram[(n + 2) % num_bins]) / 16.
    orientation_max = max(smooth_histogram)
    orientation_peaks = \
        where(logical_and(smooth_histogram > roll(smooth_histogram, 1), smooth_histogram > roll(smooth_histogram, -1)))[
            0]
    for peak_index in orientation_peaks:
        peak_value = smooth_histogram[peak_index]
        if peak_value >= peak_ratio * orientation_max:
            # Quadratic peak interpolation
            # The interpolation update is given by equation (6.30) in https://ccrma.stanford.edu/~jos/sasp/Quadratic_Interpolation_Spectral_Peaks.html
            left_value = smooth_histogram[(peak_index - 1) % num_bins]
            right_value = smooth_histogram[(peak_index + 1) % num_bins]
            interpolated_peak_index = (peak_index + 0.5 * (left_value - right_value) / (
                    left_value - 2 * peak_value + right_value)) % num_bins
            orientation = 360. - interpolated_peak_index * 360. / num_bins
            if abs(orientation - 360.) < float_tolerance:
                orientation = 0
            new_keypoint = cv2.KeyPoint(*keypoint.pt, keypoint.size, orientation, keypoint.response, keypoint.octave)
            keypoints_with_orientations.append(new_keypoint)
    return keypoints_with_orientations


def FindMaximaMinima(I):
    [h, w] = [I.shape[0], I.shape[1]]
    maxout = np.zeros((h, w))
    for i in range(1, w - 1):
        for j in range(1, h - 1):
            tp = I[j - 1:j + 2, i - 1:i + 2]
            maxima = np.max(tp)
            minima = np.min(tp)
            if maxima == tp[1, 1] or minima == tp[1, 1]:
                flag = 1
            else:
                flag = 0
            maxout[j, i] = flag
    return maxout
