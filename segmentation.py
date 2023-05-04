import numpy as np
import cv2
from scipy.spatial import KDTree


# conversion from rgb to luv
def rgb_to_luv(rgb_image):
    """
    Convert an RGB image to LUV color space.
    """
    # Define white reference values
    Xn = 0.95047
    Yn = 1.00000
    Zn = 1.09883
    
    # Normalize RGB values to [0, 1] range
    rgb_image = rgb_image.astype(np.float) / 255.0
    
    # Convert RGB to XYZ color space
    xyz_image = np.zeros(rgb_image.shape)
    xyz_image[:,:,0] = 0.4124564*rgb_image[:,:,0] + 0.3575761*rgb_image[:,:,1] + 0.1804375*rgb_image[:,:,2]
    xyz_image[:,:,1] = 0.2126729*rgb_image[:,:,0] + 0.7151522*rgb_image[:,:,1] + 0.0721750*rgb_image[:,:,2]
    xyz_image[:,:,2] = 0.0193339*rgb_image[:,:,0] + 0.1191920*rgb_image[:,:,1] + 0.9503041*rgb_image[:,:,2]
    
    # Convert XYZ to LUV color space
    luv_image = np.zeros(rgb_image.shape)
    luv_image[:,:,0] = np.where(xyz_image[:,:,1] > 0.008856, 116.0 * np.power(xyz_image[:,:,1]/Yn, 1/3.0) - 16.0, 903.3 * xyz_image[:,:,1])
    u_p = 4 * xyz_image[:,:,0] / (xyz_image[:,:,0] + 15*xyz_image[:,:,1] + 3*xyz_image[:,:,2])
    v_p = 9 * xyz_image[:,:,1] / (xyz_image[:,:,0] + 15*xyz_image[:,:,1] + 3*xyz_image[:,:,2])
    u_n = 4 * Xn / (Xn + 15*Yn + 3*Zn)
    v_n = 9 * Yn / (Xn + 15*Yn + 3*Zn)
    luv_image[:,:,1] = 13 * luv_image[:,:,0] * (u_p - u_n)
    luv_image[:,:,2] = 13 * luv_image[:,:,0] * (v_p - v_n)
    
    # Scale LUV values to [0, 255] range
    luv_image[:,:,0] = (255.0 / (np.max(luv_image[:,:,0]) - np.min(luv_image[:,:,0]))) * (luv_image[:,:,0] - np.min(luv_image[:,:,0]))
    luv_image[:,:,1] = (255.0 / (np.max(luv_image[:,:,1]) - np.min(luv_image[:,:,1]))) * (luv_image[:,:,1] - np.min(luv_image[:,:,1]))
    luv_image[:,:,2] = (255.0 / (np.max(luv_image[:,:,2]) - np.min(luv_image[:,:,2]))) * (luv_image[:,:,2] - np.min(luv_image[:,:,2]))
    
    # Convert LUV to uint8 data type
    luv_image = luv_image.astype(np.uint8)
    
    return luv_image



# kmeans implementation
def kmeans_segmentation(image, k, max_iterations=100, threshold=1e-4):
    # Convert the image into a numpy array
    img = np.array(image)
    
    # Reshape the numpy array into a 2D array
    img_shape = img.shape
    img_2d = img.reshape(img_shape[0] * img_shape[1], img_shape[2])
    
    # Initialize k centroids randomly
    centroids = img_2d[np.random.choice(img_2d.shape[0], k, replace=False)]
    
    # Assign each pixel to the closest centroid
    labels = np.zeros(img_2d.shape[0])
    distances = np.zeros(k)
    for i in range(img_2d.shape[0]):
        for j in range(k):
            distances[j] = np.linalg.norm(img_2d[i] - centroids[j])
        labels[i] = np.argmin(distances)
    
    # Update centroids based on the mean of the assigned pixels
    for i in range(k):
        centroids[i] = np.mean(img_2d[labels == i], axis=0)
    
    # Repeat the above steps until convergence
    for i in range(max_iterations):
        new_labels = np.zeros(img_2d.shape[0])
        new_distances = np.zeros(k)
        for i in range(img_2d.shape[0]):
            for j in range(k):
                new_distances[j] = np.linalg.norm(img_2d[i] - centroids[j])
            new_labels[i] = np.argmin(new_distances)
        if np.array_equal(new_labels, labels):
            break
        
        # Check if the difference between the old and new centroids is less than the threshold value
        if np.sum(np.abs(centroids - np.array([np.mean(img_2d[new_labels == i], axis=0) for i in range(k)]))) < threshold:
            break
        
        labels = new_labels
        centroids = np.array([np.mean(img_2d[labels == i], axis=0) for i in range(k)])
    
    # Reshape the labels back to the original image shape
    labels = labels.reshape(img_shape[0], img_shape[1])
    
    return labels.astype(int)




#regionGrow algorithm
class Point(object):
    def _init_(self, x, y):
        self.x = x
        self.y = y

    def getX(self):
        return self.x

    def getY(self):
        return self.y

def getGrayDiff(img, currentPoint, tmpPoint):
    return abs(int(img[currentPoint.x, currentPoint.y]) - int(img[tmpPoint.x, tmpPoint.y]))


def get_around_pixels():
    around = [Point(-1, -1), Point(0, -1), Point(1, -1),
                Point(1, 0), Point(1, 1), Point(0, 1),
                Point(-1, 1), Point(-1, 0)]
    return around

def regionGrow(img, seeds, thresh):
    height, weight = img.shape
    seedMark = np.zeros(img.shape)

    label = 1
    rest_8pixels_in_kernel = get_around_pixels()

    while (len(seeds) > 0):
        currentPoint = seeds.pop(0)

        seedMark[currentPoint.x, currentPoint.y] = label

        for i in range(8):
            neighbor_x = currentPoint.x + rest_8pixels_in_kernel[i].x
            neighbor_y = currentPoint.y + rest_8pixels_in_kernel[i].y

            if neighbor_x < 0 or neighbor_y < 0 or neighbor_x >= height or neighbor_y >= weight:
                continue

            grayDiff = getGrayDiff(img, currentPoint, Point(neighbor_x, neighbor_y))

            if grayDiff < thresh and seedMark[neighbor_x, neighbor_y] == 0:
                seedMark[neighbor_x, neighbor_y] = label
                seeds.append(Point(neighbor_x, neighbor_y))

    return seedMark

def select_random_seed(img,seeds):
    for i in range(3):
        x = np.random.randint(0, img.shape[0])
        y = np.random.randint(0, img.shape[1])
        seeds.append(Point(x, y))

def apply_region_growing(source: np.ndarray):

    src = np.copy(source)
    color_img = cv2.cvtColor(src, cv2.COLOR_Luv2BGR)
    img_gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    global seeds
    seeds = []   
    select_random_seed(img_gray,seeds)
    output_image = regionGrow(img_gray, seeds, 10)

    return output_image


# Define the region growing function:
# """
# this algorithm need from user to put values for thersholding points and the location of each seed point, the user can add point as he
#  want and see how the region change every time he change even one of them or changing the both seed points and its thresholds  

# """

# def region_grow(image, seeds, threshold):
#     # Get the dimensions of the image
#     rows, cols, channels = image.shape

#     # Create a binary mask to keep track of the pixels in the region
#     mask = np.zeros((rows, cols), dtype=np.uint8)

#     # Convert the image to LUV color space
#     luv_image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)

#     # Get the L, U, and V channels
#     L, U, V = cv2.split(luv_image)

#     # Define the neighbors of a pixel
#     neighbors = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])

#     # Initialize the queue with the seed pixels
#     queue = list(seeds)

#     # Process the queue until it is empty
#     while len(queue) > 0:
#         # Get the first pixel from the queue
#         pixel = queue.pop(0)

#         # Check if the pixel has already been added to the region
#         if mask[pixel[0], pixel[1]] == 1:
#             continue

#         # Get the intensity values of the pixel
#         intensity_L = float(L[pixel[0], pixel[1]])
#         intensity_U = float(U[pixel[0], pixel[1]])
#         intensity_V = float(V[pixel[0], pixel[1]])

#         # Check if the pixel is similar enough to any of the seed pixels
#         for seed in seeds:
#             seed_L = float(L[seed[0], seed[1]])
#             seed_U = float(U[seed[0], seed[1]])
#             seed_V = float(V[seed[0], seed[1]])
#             if abs(intensity_L - seed_L) <= threshold[0] and abs(intensity_U - seed_U) <= threshold[1] and abs(intensity_V - seed_V) <= threshold[2]:
#                 # Add the pixel to the region
#                 mask[pixel[0], pixel[1]] = 1

#                 # Add the neighbors of the pixel to the queue
#                 for neighbor in neighbors:
#                     neighbor_pixel = pixel + neighbor
#                     if neighbor_pixel[0] < 0 or neighbor_pixel[0] >= rows or neighbor_pixel[1] < 0 or neighbor_pixel[1] >= cols:
#                         continue
#                     queue.append(neighbor_pixel)


#     return mask




# Mean Shift
def mean_shift(img, window_size=30, criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1)):
    # Reshape the image into a 2D array of pixels
    img_to_2dArray = img.reshape(-1, 3)

    num_points, num_features = img_to_2dArray.shape
    point_considered = np.zeros(num_points, dtype=bool)
    labels = -1 * np.ones(num_points, dtype=int)
    label_count = 0

    # Use a KD-tree to efficiently find the points within the window
    tree = KDTree(img_to_2dArray)

    for i in range(num_points):
        if point_considered[i]:
            continue

        Center_point = img_to_2dArray[i]
        while True:
            # Find all points within the window centered on the current point
            in_window = tree.query_ball_point(Center_point, r=window_size)

            # Calculate the mean of the points within the window
            new_center = np.mean(img_to_2dArray[in_window], axis=0)

            # If the center has converged, assign labels to all points in the window
            if np.linalg.norm(new_center - Center_point) < criteria[1]:
                labels[in_window] = label_count
                point_considered[in_window] = True
                label_count += 1
                break

            Center_point = new_center

    labels = labels.reshape(img.shape[:2])  

    # Create a new image where each pixel is assigned the color of its cluster centroid
    new_img = np.zeros_like(img)
    for i in range(np.max(labels)+1):
        new_img[labels == i] = np.mean(img[labels == i], axis=0)

    output_image = np.array(new_img, np.uint8)
    return output_image




#Agglomerative algorithm
def calculate_distance(x1, x2):
    """
    Calculates Euclidean distance between two points.
    """
    return np.sqrt(np.sum((x1 - x2) ** 2))

def clusters_distance(cluster1, cluster2):
    """
    Calculates the distance between two clusters as the maximum distance between any two points in the clusters.
    """
    return max([calculate_distance(point1, point2) for point1 in cluster1 for point2 in cluster2])

def clusters_mean_distance(cluster1, cluster2):
    """
    Calculates the mean distance between two clusters as the Euclidean distance between their centroids.
    """
    cluster1_center = np.average(cluster1, axis=0)
    cluster2_center = np.average(cluster2, axis=0)
    return calculate_distance(cluster1_center, cluster2_center)

def initial_clusters(image_clusters, k):
    """
    Initializes the clusters with k colors by grouping the image pixels based on their color.
    """
    cluster_color = int(256 / k)
    groups = {}
    for i in range(k):
        color = i * cluster_color
        groups[(color, color, color)] = []
    for i, p in enumerate(image_clusters):
        go = min(groups.keys(), key=lambda c: calculate_distance(p, c))
        groups[go].append(p)
    return [group for group in groups.values() if len(group) > 0]

def get_cluster_center(point, cluster, centers):
    """
    Returns the center of the cluster to which the given point belongs.
    """
    point_cluster_num = cluster[tuple(point)]
    center = centers[point_cluster_num]
    return center

def get_clusters(image_clusters, clusters_number):
    """
    Agglomerative clustering algorithm to group the image pixels into a specified number of clusters.
    """
    clusters_list = initial_clusters(image_clusters, clusters_number)
    cluster = {}
    centers = {}

    while len(clusters_list) > clusters_number:
        cluster1, cluster2 = min(
            [(c1, c2) for i, c1 in enumerate(clusters_list) for c2 in clusters_list[:i]],
            key=lambda c: clusters_mean_distance(c[0], c[1]))

        clusters_list = [cluster_itr for cluster_itr in clusters_list if not np.array_equal(cluster_itr, cluster1) and not np.array_equal(cluster_itr, cluster2)]
        merged_cluster = cluster1 + cluster2
        clusters_list.append(merged_cluster)

    for cl_num, cl in enumerate(clusters_list):
        for point in cl:
            cluster[tuple(point)] = cl_num

    for cl_num, cl in enumerate(clusters_list):
        centers[cl_num] = np.average(cl, axis=0)

    return cluster, centers

def apply_agglomerative_clustering(image, clusters_number, initial_clusters_number):
    """
    Applies agglomerative clustering to the image and returns the segmented image.
    """
    flattened_image = np.copy(image.reshape((-1, 3)))
    cluster, centers = get_clusters(flattened_image, clusters_number)
    output_image = []
    for row in image:
        rows = []
        for col in row:
            rows.append(get_cluster_center(list(col), cluster, centers))
        output_image.append(rows)    
    output_image = np.array(output_image, np.uint8)
    return output_image