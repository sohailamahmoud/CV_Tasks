# Computer-Vision Tasks


# CV-A01: Images, Filters, Histograms, Gradients, Frequency

## Objectives
- Filtration of noisy images using low pass filters such as: average, Gaussian, median.
- Edge detection using variety of masks such as: Sobel, Prewitt, and canny edge detectors.
- Histograms and equalization.
- Frequency domain filters.
- Hybrid images.

## Deadline
**Wednesday 8 Mar 2023 at 11:59 PM**

## Deliverables
- Filters.py : this will include you implementation for filtration functions (requirements 1-3)
- Frequency.py : this will include your implementation for histogram related tasks (requirements 4-8).
- Histograms.py : this will include your implementation for frequency domain related tasks (requirements 9-10).
- Images : folder contains images to test your implementation.
- UI necessary files

## GUI
Create three tabs:
- Tab 1 : load and show input image, choose a filter from list of available filters, determine filter parameters, apply the fiter then show the output image.
- Tab 2 : load and show input image, calculate and plot input image histogram, apply histogram equalization and show output image, plot output histogram.
- Tab 3 : load input image A, load input image B, make and show hybrid image.


# CV-A02
## Objectives
* Apply Hough transform for detecting parametric shapes like circles and lines
* Apply Active Contour Model for semi-supervised shape delineation.


## Three python files 
* **Hough.py**: includes implementation for Hough transform for lines, circles, and Ellipses.
* **ActiveContour.py**: includes implementation for Active Contour Model for semi-supervised shape delineation and the ChainCode  of the Contour.
* **App.py**: The main file for GUI using Streamlit open-source framework.


# Implemented Functions

## Hough-Line-Detection
Implementation of Simple Hough Line Detection Algorithm in Python.

### Input
The script requires one positional argument and few optional parameters:
* image_path : Complete path to the image file for circle detection.
* bin_threshold : Thresholding value . Default is 150.
* min_edge_threshold : Minimum threshold value for edge detection. Default 100.
* max_edge_threshold : Maximum threshold value for edge detection. Default 200.

### Output

![line](https://user-images.githubusercontent.com/93046966/227671967-7a7530a4-7c08-4c2a-b3b5-9a65fb58be85.png)



## Hough-Circle-Detection
Implementation of Simple Hough Circle Detection Algorithm in Python.

### Input
The script requires one positional argument and few optional parameters:
* image_path : Complete path to the image file for circle detection.
* bin_threshold : Thresholding value . Default is 150.
* min_edge_threshold : Minimum threshold value for edge detection. Default 100.
* max_edge_threshold : Maximum threshold value for edge detection. Default 200.

### Output

![circle](https://user-images.githubusercontent.com/93046966/227671683-02171c70-83f1-4540-9ed4-20fdca20a016.png)



## Hough-Ellipse-Detection
Implementation of Simple Hough Circle Detection Algorithm in Python.

### Input
The script requires one positional argument and few optional parameters:
* image_path : Complete path to the image file for circle detection.
* bin_threshold : Thresholding value . Default is 150.
* min_edge_threshold : Minimum threshold value for edge detection. Default 100.
* max_edge_threshold : Maximum threshold value for edge detection. Default 200.

### Output
![Ellipse](https://user-images.githubusercontent.com/93046966/227672233-60740bf8-5bbe-42ec-8d34-ef422c826c3b.png)



## Active-Contour
Implementation of Simple Active Contour Snake Algorithm in Python.

### Input
The script requires two positional arguments and few optional parameters:
* image : ND array of input image in gray-scale after applying guassian filter.
* snake : ND array of points of the inital snake.
* alpha : weight of elasticity component of snake. Default 0.01.
* beta : weight of stiffness component of snake. Default 0.1.
* w_line : weight of attraction force to brightness of input image. Default 0.0.
* w_edge : weight of attraction to edges of input image. Default 1.0.
* gamma : weight of energy that forces the snake toward or away from edges. Default = 0.01.
* max_px_move : Maximum pixel distance to move per iteration. Default = 1.0.
* max_num_iter : Maximum iterations to optimize snake shape. Default = 2500.

### Output
![active_contour](https://user-images.githubusercontent.com/81927516/227679157-dd6f64f0-e5ac-4e7f-92a2-5b25af92f97c.png)



## Chain Code
Implementation of Simple Chain Code calculation for Snake Algorithm in Python.

### Input
The script requires one positional argument:
* snake : array of the final snake calculated.

### Output
![chaincode](https://user-images.githubusercontent.com/81927516/227679958-ef3e8d80-41ab-4f5d-ab71-83dbbdeb153c.png)

# CV-A03
## Objectives
* Harries Algorithm
* SIFT Algorithm
* Feature Matching

## Harries Algorithm

The Harris corner detection function is used for identifying corners and inferring features of an image. Corners are the important features in the image, which are invariant to translation, rotation, and illumination.

* Original Image

![Original Image](https://user-images.githubusercontent.com/37380802/231598285-bb96a0c2-f536-4da7-91c2-421d56b88e98.PNG)

* After Applying Harris Algorithm

![After Applying Harris Algorithm](https://user-images.githubusercontent.com/37380802/231598595-dfa3d487-1fd0-4160-bc69-5861a0cca1d1.PNG)

* UI Apperrance

![image](https://user-images.githubusercontent.com/37380802/231603173-5127760a-f5bd-4d42-8d33-beb24115c5fa.png)


## SIFT Algorithm

SIFT Scale-Invariant Feature Transform (SIFT) is another technique helps locate the local features in an image, commonly known as the ‘keypoints‘ of the image. These keypoints are scale & rotation invariants that can be used for various computer vision applications.

* Original Image

![image](https://user-images.githubusercontent.com/37380802/231601912-f4e8c6bb-c26d-4061-b57d-6cb41e981451.png)

* After Applying SIFT

![image](https://user-images.githubusercontent.com/37380802/231602024-a35541a7-871e-4b3b-a0f9-86842f9c1dff.png)

## Feature Matching
* Using SSD
  
![SSD](https://user-images.githubusercontent.com/37380802/231600979-1da9e9ef-0883-48c7-a1b8-2ad7f90725b7.jpeg)
  
* Using Normalized Correlation (NCC)
 
![NCC](https://user-images.githubusercontent.com/37380802/231601282-cb53cc4c-99d2-481d-8b97-f6cd380890cd.jpeg)

* UI Apperrance

![image](https://user-images.githubusercontent.com/37380802/231603915-32b6880a-9352-4430-95e9-9c45bf4b8699.png)

# CV-A04
## Objectives
* Thresholding (optimal, Otsu,spectral)
* Unspervised segmentation using k-means, region growing, agglomerative and mean shift methods.

## Optimal Thresholding
method used to automatically determine the threshold value to separate an image into foreground and background regions.
* Original Image

![image](https://user-images.githubusercontent.com/37380802/236079634-eb4bd1fd-f301-411a-9de0-8a3856e621fe.png)

* Optimal Global                                 

![image](https://user-images.githubusercontent.com/37380802/236079729-93b9523c-5594-4b2a-9341-1d723f13fe5a.png)

* Optimal Local 

 ![image](https://user-images.githubusercontent.com/37380802/236079952-ebe37184-5f94-4d37-9542-e2a8ace29c02.png)
 
* UI Apperance

![image](https://user-images.githubusercontent.com/37380802/236088890-67b796f5-3e73-45e4-9283-917b1a633ae5.png)

## Otsu Thresholding
a widely used image thresholding method that automatically determines the threshold value for image segmentation.
* Original Otsu

![image](https://user-images.githubusercontent.com/37380802/236080450-1eb35b27-e5ec-42ef-a5fe-60a14fb82d4c.png)

* Otsu Global                                 

![image](https://user-images.githubusercontent.com/37380802/236080561-ca354599-5e44-499c-b3fe-4b0ecb13cdf2.png)

* Otsu Local 

![image](https://user-images.githubusercontent.com/37380802/236080669-5430e03f-7f3e-41fa-8c24-bff52a81fb66.png)

* UI Apperrance

![image](https://user-images.githubusercontent.com/37380802/236080960-4e5fd6d2-3bbb-4517-a6c2-70c6f3421765.png)

## Spectral Thresholding
a method of image segmentation that involves identifying different modes or clusters in an image's intensity histogram and assigning each cluster to a different segment or class
* Original Image

![image](https://user-images.githubusercontent.com/37380802/236081719-a6ca58af-2067-4036-80fd-2ee1d147d3a6.png)

* Spectral Global                                 

![image](https://user-images.githubusercontent.com/37380802/236082060-50ee87f0-eddd-4bd7-ba83-5a90c7f94516.png)

* Spectral Local 

![image](https://user-images.githubusercontent.com/37380802/236082179-fe53509e-7183-4659-9ce6-b623599a0399.png)

* UI Apperrance

![spectral](https://user-images.githubusercontent.com/81927516/236093458-5965a28c-e84e-4648-92d8-60cf925ec4a9.png)



## RGB to LUV Conversion
algorithm for converting an image from RGB scale to LUV scale.

* Output Image

![image](https://user-images.githubusercontent.com/81927516/236095468-0e032e86-8db8-497d-8add-7e77edba4014.png)

* UI Appearance

![luv](https://user-images.githubusercontent.com/81927516/236095728-90fcfe19-9095-4246-98a0-97a8b3ed0f9b.png)


## K-means Segmentation
K-means segmentation is a clustering algorithm that partitions an image into K clusters, where K is a user-defined number.

* Original Image

![image](https://user-images.githubusercontent.com/37380802/236083164-d867c47a-cb99-4b18-b5ad-b557190427aa.png)

* Segmented Image (k=5)                               

![image](https://user-images.githubusercontent.com/37380802/236083293-6b115bf4-7c8f-453f-bb2b-f8eb0eef5705.png)

* Segmented Image (k=7)  

![image](https://user-images.githubusercontent.com/37380802/236083367-3f3e9381-150a-44c7-8fc6-07e39d717287.png)

## Region Growing Segmentation
Region growing is a segmentation algorithm that groups pixels in an image based on their similarity to each other.

* Original Image

![image](https://user-images.githubusercontent.com/37380802/236083985-663141e1-2d72-4f16-9f13-f550ab3b9046.png)

* With Random Seed                            

![image](https://user-images.githubusercontent.com/37380802/236084157-e9faa015-7f24-4653-808a-f0691f90484b.png)

* With Another Random Seed 

 ![image](https://user-images.githubusercontent.com/37380802/236084328-a9b9ed50-ab0c-46ab-9cfd-4fd6e14cee58.png)
 
 * With set seed points and threshold seeds = [(70, 70), (100, 150), (200, 200)] and threshold = (10, 55, 105)

 ![image](https://user-images.githubusercontent.com/37380802/236084510-f9a3e0ee-a830-412a-8d51-bbd9111bce87.png)
 
 *	With another set seed points and threshold:
  seeds = [(70, 70), (100, 150), (200, 200)] and threshold = (30, 55, 105)
  
  ![image](https://user-images.githubusercontent.com/37380802/236085090-96d89b08-da6a-4f9a-9efc-0957023699ea.png)

## Agglomerative Segmentation
Agglomerative segmentation is a technique that groups similar pixels into clusters based on their proximity to each other. The algorithm starts by considering each pixel as a separate cluster and then merges the closest clusters iteratively until a desired number of segments is achieved.

* Original Image
 
 ![image](https://user-images.githubusercontent.com/37380802/236085414-acfeac27-6796-4aa2-9a53-963960b109be.png)

* With number of clusters equal 5

![image](https://user-images.githubusercontent.com/37380802/236085535-3d3b4da2-56b1-49cd-9d0f-0780ff4a6f55.png)

*	With number of clusters equal 10

![image](https://user-images.githubusercontent.com/37380802/236085972-6f046641-87df-4d7c-81b1-166037df6acf.png)

* With number of clusters equal 25

![image](https://user-images.githubusercontent.com/37380802/236086656-36e09a19-7236-4394-ba61-d18f94d2f21f.png)

* UI Apperance 

![agglomerative](https://user-images.githubusercontent.com/81927516/236093742-86d3412b-56bb-4d93-81c0-c3112091a58f.png)



## Mean Shift Segmentation
Mean shift segmentation is a clustering-based algorithm that is used to group similar pixels in an image together into distinct regions. The algorithm works by first selecting a window around each pixel in the image. The size of the window is determined by the user, which controls the sensitivity of the algorithm to changes in pixel intensity.

* Original Image

 ![image](https://user-images.githubusercontent.com/37380802/236087146-978272f7-79e6-4f7d-8d68-6f78a9cff9fb.png)

* Mean Shift with window size 60

![image](https://user-images.githubusercontent.com/37380802/236087516-31b2d17d-e1f2-4f03-a163-2f8f02a8baef.png)

* Mean Shift with window size 30

![image](https://user-images.githubusercontent.com/37380802/236087809-794f37fb-f9e5-4efd-b4ca-6d8fa3268ced.png)

* Mean Shift with window size 60
 
 ![image](https://user-images.githubusercontent.com/37380802/236088058-0725e070-54d8-497f-882b-74ff220f5b5b.png)

* Mean Shift with window size 90

![image](https://user-images.githubusercontent.com/37380802/236088333-82940544-fdaa-4fad-b394-f2775701d078.png)

* UI Apperance 

![mean_shift](https://user-images.githubusercontent.com/81927516/236093838-add00e75-a29d-41cb-87f1-85e58dcb7d01.png)


## About Us

### Team  Number 8 

### Members

Name| Section | Bench Number |
--- | --- | --- |
[Maryam Megahed](https://github.com/MaryamMegahed "Maryam Megahed") | 2 | 32
[Arwa Essam](https://github.com/arwa-essam "Arwa Essam") | 1 | 10
[Shrouk Shawky](https://github.com/shirouq-shawky "Shrouk Shawky") | 1 | 46
[Sohaila Mahmoud](https://github.com/sohailamahmoud "Sohaila Mahmoud") | 1 | 45
