import streamlit as st
import cv2
import time
import Hough , ActiveContour, Sift , MatchingFeatures , harrisDetector
from faceDetectRecognition import * 
from segmentation import *
from thresholding import *
from filter import *
from histogram import *
from frequency import *
import numpy as np
import seaborn as sns
from skimage import io
from matplotlib import pyplot as plt
import logging
from skimage.filters import gaussian
import matplotlib.pyplot as plt
from skimage.feature import canny
from skimage.draw import ellipse_perimeter






#page styling
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(page_title='Computer Vision', layout="wide")

reduce_header_height_style = """
    <style>
        div.block-container {padding-top:0rem;}
    </style>
"""
st.markdown(reduce_header_height_style, unsafe_allow_html=True)


st.write(
    '<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: left;} </style>', unsafe_allow_html=True)

hide_st_style = """
            <style>
            # MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
local_css("style.css")



tab1, tab2,tab3,tab4,tab5,tab6,tab7, tab8 , tab9, tab10, tab11 = st.tabs(["Filters", "Histogram", "Frequency","Shape Detections", "Contour","Harris corners","Features Matching","Thresholding","Segmentation","Face Detection", "Face Recognition"])


with st.sidebar:
    main_image= st.file_uploader ("Upload Image ", type= ["jpg","png","bmp","jpeg"], accept_multiple_files=False, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")

    with st.expander("Filter and Edge Detection", expanded=False):
        btn3= st.selectbox('Edge detection ',('Sobel', 'Roberts', 'Prewitt' ,'Canny'))
        btn1= st.selectbox('Add noise to image',('Uniform', 'Gaussian', 'salt & pepper'))
        btn2= st.selectbox('Filter Image',('Average', 'Gaussian', 'median'))
        btn_filter=st.selectbox("choose size",('3','5'))
    
    with st.expander("Histogram",expanded=False):
            btn5=st.selectbox("Thresholding",options=('local','global'))
            btn4=st.selectbox("Processing",options=('Equalization','Normalization')) 

    with st.expander("Detection Options",expanded=False):
        btn6= st.selectbox(' Detection Type',('Lines', 'Circles' ,'Ellipses'))

        if main_image:
            lower_threshold= st.slider("Lower Edge Threshold",min_value= 0,max_value= 255,value= 100 ,step=1)
            upper_threshold= st.slider("Upper Edge Threshold",min_value= 0,max_value= 255,value= 200 ,step=1)

            if(btn6=='Lines' or btn6 == 'Circles'):
                threshold= st.slider("Threshold",min_value= 0,max_value= 400,value= 150 ,step=1)
    

    with st.expander("Active Contour",expanded=False):
             # user inputs for active contour paramaters
            alpha = st.number_input('Alpha',min_value=0.0, max_value=10.0, value=0.1, step=0.1)
            beta = st.number_input('Beta',min_value=0.0, max_value=10.0, value=0.1, step=0.1)
            gamma = st.number_input('Gamma',min_value=0.0, max_value=10.0, value=0.1, step=0.1)
            w_line = st.number_input('w-line',min_value=-10, max_value=10, value=0, step=1)
            w_edge = st.number_input('w-edge',min_value=-10, max_value=10, value=1, step=1)


    with st.expander("Thresholding", expanded=False):
        btn_threshold= st.selectbox('Thresholding Option',('Optimal', 'Otsu', 'Spectral'))

    
    with st.expander("Segmentation", expanded=False):
        btn_segment= st.selectbox('Segmentation Option',('K-means', 'Region Growing', 'Agglomerative', 'Mean Shift'))

        if main_image:
            if (btn_segment=='K-means'):
                cluster_number = st.number_input('Number of Clusters',min_value=1, max_value=100, value=5, step=1)
                iteration_number = st.number_input('Number of Maximum Iterations',min_value=1, max_value=3000, value=50, step=1)
            
            if (btn_segment=='Agglomerative'):
                initial_cluster = st.number_input('Number of Initial Clusters',min_value=1, max_value=100, value=25, step=1)
                final_cluster = st.number_input('Number of Final Clusters',min_value=1, max_value=100, value=10, step=1)
    
    with st.expander("Face Detection",expanded= False):
        scale_factor = st.number_input('Scale Factor',min_value=0.0, max_value=10.0, value=1.4, step=0.1)
        min_neighbour = st.number_input('minimum neighbor',min_value=1, max_value=100, value=5, step=1)
    
    with st.expander("Face Recognition",expanded= False):
        width = st.number_input('Width',min_value=1, max_value=200, value=64, step=1)
        height = st.number_input('Height',min_value=1, max_value=200, value=64, step=1)
        Euclidean_threshold = st.number_input('Threshold for the accepted Euclidean Distance',min_value=1, max_value=1000, value=200, step=1)
    
        
    

#filters tab
with tab1:
       
    with st.container():
        col1,a,col2,b,col3,c,col4 = st.columns([2,0.2,2,0.2,2,0.2,2]) 
        if main_image:
            path = "images/" + main_image.name
            image= cv2.imread(path,cv2.IMREAD_GRAYSCALE) 
            with col1:
                st.image(image, caption='Original image' ,width=300) 
                    
            with col2:
                # Blur the image for better edge detection
                img_blur = cv2.GaussianBlur(image, (3,3), 0)
                if(btn3=='Sobel'):
                    img_edge_detected=SobelXY(img_blur)
                    st.image(img_edge_detected, caption='edge detected image' ,width=300,clamp=True)  
                elif(btn3=='Roberts'):
                    img_edge_detected=Robert(image)
                    st.image(img_edge_detected, caption='edge detected image' ,width=300,clamp=True)   
                elif(btn3=='Prewitt'):
                    img_edge_detected=prewitt(img_blur)
                    st.image(img_edge_detected, caption='edge detected image' ,width=300,clamp=True) 
                elif(btn3=='Canny'):
                    img_edge_detected=cannyDetection(img_blur)
                    st.image(img_edge_detected, caption='edge detected image' ,width=300,clamp=True) 
                                  
            with col3:    
                image= cv2.imread(path,cv2.IMREAD_GRAYSCALE) 
                if(btn1=='Uniform'):
                    img_noised=Addnoise(image,'Uniform')
                    #st.image(img_noised, caption='noised image' ,width=300,clamp=True) 
                    st.image("images\\result_image.bmp", caption='noised image' ,width=300)
                elif(btn1=='Gaussian'):
                    img_noised=Addnoise(image,'Gaussian')
                    #st.image(img_noised, caption='noised image' ,width=300,clamp=True) 
                    st.image("images\\noisy.bmp", caption='noised image' ,width=300)
                elif(btn1=='salt & pepper'):
                    img_noised=Addnoise(image,'salt & pepper')
                    st.image(img_noised, caption='noised image' ,width=300,clamp=True)  
          
            with col4:
                if(btn1=='Uniform'): 
                    noise=cv2.imread("images\\result_image.bmp",cv2.IMREAD_GRAYSCALE)
                elif(btn1=='Gaussian'): 
                    noise=cv2.imread("images\\noisy.bmp",cv2.IMREAD_GRAYSCALE)
                elif(btn1=='salt & pepper'):
                    noise = img_noised
                    
                if(btn2=='Average'):
                    if(btn_filter=='3'):
                        img_filtered=MeanFilter(noise, 9)
                        st.image(img_filtered, caption='filtered image' ,width=300,clamp=True) 
                    elif(btn_filter=='5'):
                        img_filtered=MeanFilter(noise, 25)
                        st.image(img_filtered, caption='filtered image' ,width=300,clamp=True) 
                    
                elif(btn2=='Gaussian'):
                    if(btn_filter=='3'):
                        img_filtered=gaussian_filter(noise, 3, sigma=1)
                        st.image(img_filtered, caption='filtered image' ,width=300,clamp=True)
                    elif(btn_filter=='5'):
                        img_filtered=gaussian_filter(noise, 5, sigma=1)
                        st.image(img_filtered, caption='filtered image' ,width=300,clamp=True)
                  
                elif(btn2=='median'):
                    if(btn_filter=='3'):
                        img_filtered=median_filter(noise, 3)
                        st.image(img_filtered, caption='filtered image' ,width=300,clamp=True) 
                    elif(btn_filter=='5'):    
                        img_filtered=median_filter(noise, 5)
                        st.image(img_filtered, caption='filtered image' ,width=300,clamp=True) 




#histogram tab                              
with tab2:
    

    with st.container():
        col1,col2,col3 = st.columns([2,2,2])
        with col1:
           if main_image:
                path1 = "images/" + main_image.name
                im2=cv2.imread(path1,cv2.IMREAD_GRAYSCALE) 
                st.image(im2, caption='Original image' ,width=300)
                 
        with col2:
            if(main_image):
                path1 = "images/" + main_image.name
                im2=cv2.imread(path1,cv2.IMREAD_GRAYSCALE) 
                if(btn4=='Equalization'):
                    equalized_image=equalization(im2,256)
                    st.image(equalized_image, caption='Equalized image' ,width=300) 
                if(btn4=='Normalization'):
                    normalize(img=im2)
                    st.image("images\\normalized.bmp", caption='Normalized image' ,width=300)    
                    
        with col3:       
            if(main_image):
                if(btn5=='local'):
                    Local=localThresholding(im2,60,60,60,60)
                    st.image(Local, caption='Local Threshold image' ,width=300) 
                elif(btn5=='global'):
                    Global= global_thresholding(im2,120)
                    st.image(Global, caption='Global Thresholding image' ,width=300)


    with st.container():
        col1,col2 = st.columns([2,2])
        if(main_image):
            with col1:
                image= cv2.imread(path1,cv2.IMREAD_GRAYSCALE)
                sns.distplot(image, color="grey", label="Density")
                plt.title('Grayscale Histogram')
                plt.xlabel('Intensity Value')
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.pyplot()
            with col2:
                im= io.imread(path1)
                RGB_histogram = plt.hist(im[:, :, 0].ravel(), bins = 256, color = 'red')
                RGB_histogram = plt.hist(im[:, :, 1].ravel(), bins = 256, color = 'Green')
                RGB_histogram = plt.hist(im[:, :, 2].ravel(), bins = 256, color = 'Blue')
                plt.title('RGB Histograms')
                RGB_histogram = plt.xlabel('Intensity Value')
                RGB_histogram = plt.ylabel('Count')
                RGB_histogram = plt.legend([ 'Red_Channel', 'Green_Channel', 'Blue_Channel'])   
                st.pyplot()     
        
                
    with st.container():
        col1,col2,col3=st.columns([2,2,2])
        if(main_image):
            im= io.imread(path1)
            with col1:
                R=im[:, :, 0]
                R_Distribution = sns.distplot(R, color="red", label="Compact")
                R_Distribution= plt.xlabel('Intensity Value')
                plt.title('R Distribution')
                st.pyplot()
                R_Cumulative = plt.hist(R.ravel(), bins = 256, cumulative = True,color="red")
                R_Cumulative= plt.xlabel('Intensity Value')
                R_Cumulative= plt.ylabel('Count') 
                plt.title('R Cumulative')
                st.pyplot()
            with col2:    
                G=im[:, :, 1]
                G_Distribution= sns.distplot(G, color="green", label="Compact")
                G_Distribution= plt.xlabel('Intensity Value')
                plt.title('G Distribution')
                st.pyplot()
                G_Cumulative= plt.hist(G.ravel(), bins = 256, cumulative = True,color="green")
                G_Cumulative= plt.xlabel('Intensity Value')
                G_Cumulative= plt.ylabel('Count') 
                plt.title('G Cumulative')
                st.pyplot()
            with col3:
                B=im[:, :, 2]
                B_Distribution = sns.distplot(B, color="blue", label="Compact")
                B_Distribution= plt.xlabel('Intensity Value')
                plt.title('B Distribution')
                st.pyplot()
                B_Cumulative = plt.hist(B.ravel(), bins = 256, cumulative = True,color="blue")
                B_Cumulative= plt.xlabel('Intensity Value')
                B_Cumulative= plt.ylabel('Count') 
                plt.title('B Cumulative')
                st.pyplot()






#hybrid image tab
with tab3:
  with st.container():
    col1,c,col2,c,col3 = st.columns([2,0.3,2,0.2,1.5])
    with col1:
        img1= st.file_uploader ("Upload Image 1", type= ["jpg","png","bmp","jpeg"], accept_multiple_files=False, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
    with col2:
        img2= st.file_uploader ("Upload Image 2", type= ["jpg","png","bmp","jpeg"], accept_multiple_files=False, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
    with col3:
        x= st.button("Generate hybrid image") 
            
            
    with st.container():
        col1,col2, col3 = st.columns([2,2,2])
        with col1:
           if img1:
                path1 = "images/" + img1.name
                image1 = cv2.imread(path1,0) 
                image1 = image1.astype(np.float32)/255
                lowpass =LowPass(image1)
                st.image(lowpass, caption='Low Pass Filtered image' ,width=350,clamp=True) 
        with col2:
             if img2:
                path2 = "images/" + img2.name
                image2 = cv2.imread(path2,0) 
                image2 = image2.astype(np.float32)/255
                highpass=HighPass(image2)
                st.image(highpass, caption='High Pass Filtered image' ,width=350,clamp=True) 
        with col3:
            if (img1 and img2 and x):
                hy_img=hybrid(lowpass,highpass)
                st.image(hy_img, caption='Hybrid image' ,width=350,clamp=True)





#shapes detection tab
with tab4:

    with st.container():
        col1,a,col2,b,col3 = st.columns([2,0.2,2,0.2,2]) 
        if main_image:
            path = "images/" + main_image.name
            orig_img = cv2.imread(path,cv2.IMREAD_COLOR)
            input = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
            smoothed_img = Hough.gaussian_smoothing(input_img=input)
            with col1:
                st.image(path, caption='Original image' ,width=300) 

            with col2:
                edged_image = Hough.canny_edge_detection(smoothed_img,lower_threshold,upper_threshold)
                st.image("images\edge_Detected_Image.jpg", caption='Edged image' ,width=300)

            with col3:
                
                if(btn6=='Lines'):
                    Hough.Hough_line_detection(path=path,l_threshold=lower_threshold,u_threshold=upper_threshold,threshold=threshold)
                    st.image("images/Detected_Line_Image.jpg", caption='Line detected image' ,width=300,clamp=True) 



                elif(btn6=='Circles'):
                    Hough.Hough_circle_detection(path=path,threshold=threshold,l_threshold=lower_threshold,u_threshold=upper_threshold)
                    st.image("images\Circle_Detected_Image.jpg", caption='circle detected image' ,width=300,clamp=True) 
                
                
                elif(btn6=='Ellipses'):
                    image_gray = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
                    edges = canny(image_gray, sigma=2.0,
                                low_threshold=0.55, high_threshold=0.8)
                    result = Hough.hough_ellipse(edges, accuracy=20, threshold=250,
                                        min_size=100, max_size=120)
                    result.sort(order='accumulator')

                    # Estimated parameters for the ellipse
                    best = list(result[-1])
                    yc, xc, a, b = (int(round(x)) for x in best[1:5])
                    orientation = best[5]

                    # Draw the ellipse on the original image with thicker lines
                    for offset in range(-2, 3):
                        cy, cx = ellipse_perimeter(yc + offset, xc + offset, a, b, orientation)
                        orig_img[cy, cx] = (0, 0, 255)

                    # Display the image with the thicker ellipse
                    cv2.imwrite("images\Ellipse_Detected_Image.jpg",orig_img,)
                    st.image("images\Ellipse_Detected_Image.jpg",caption="Detected Ellipse",width=300)
                    






#active contour  and chaincode tab                              
with tab5:
    

    #Active contour
    with st.container():
        col1,col2 = st.columns([2,2])
        if main_image:
            path1 = "images/" + main_image.name
            im2=cv2.imread(path1,cv2.COLOR_BGR2RGB) 
            img= cv2.imread(path1, cv2.IMREAD_GRAYSCALE)

        with col1:
            if(main_image):
                st.image(path1, caption='Original image' ,width=300)


        with col2:
            if(main_image):
                path1 = "images/" + main_image.name
                
                #initializing first contour points
                s = np.linspace(0, 2*np.pi, 400)
                x = 100 + 100*np.cos(s)
                y = 100 + 100*np.sin(s)
                init = np.array([x, y]).T
                
                #final contour points
                snake = ActiveContour.ActiveContourSnake(gaussian(img, 3),
                        init, alpha=alpha, beta=beta, gamma=gamma,w_edge=w_edge,w_line=w_line)
                
                #displaying active contour on image
                fig = plt.figure(figsize=(7, 7))
                ax = fig.add_subplot(111)
                plt.gray()
                ax.imshow(img)
                ax.plot(init[:, 0], init[:, 1], '--r', lw=3)
                ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3)
                # ax.set_xticks([]), ax.set_yticks([])
                ax.axis([0, img.shape[1], img.shape[0], 0])
                plt.savefig("images/contoured_image.png")
                st.image("images/contoured_image.png", caption="Contoured Image", width=300)


    #chaincode, perimeter and area calculations            
    with st.container():

        if(main_image):
            with st.sidebar:

                st.subheader("Active Contour Area and Perimeter")

                #calculating chaincode of the final contour
                chaincode = ActiveContour.ChainCode(snake= snake)
                print("Chaincode of the contour",chaincode)

                #calculating area of the final contour
                area = ActiveContour.areaOfContour(snake=snake)
                col1,col2= st.columns([1,1])
                with col1:
                    st.markdown("Area = ")
                with col2:
                    st.write(area)
                
                #calculating perimeter of the final contour
                perimeter = ActiveContour.perimeter(snake=snake)
                col1,col2 = st.columns([1,1])
                with col1:
                    st.markdown("Perimeter= ")
                with col2:
                    st.write(perimeter)




#harris tab
with tab6:
    col1,a,col2 = st.columns([2,0.2,2])
    if main_image:
        path = "images/" + main_image.name
        image= cv2.imread(path,cv2.IMREAD_GRAYSCALE) 
        
        with col1:
            st.image(image, caption='Original image' ,width=300) 

        with col2:
            harris , execution_time = harrisDetector.harrisCorner(img=image)
            st.image(harris, caption='Corner Image' ,width=350,clamp=True)
            print("Execution time for harris detector: ", execution_time )


# Image Matching tab
with tab7:
    with st.container():
        col1,c,col2= st.columns([2,0.3,2])
        with col1:
            img1= st.file_uploader ("Upload Image", type= ["jpg","png","bmp","jpeg"], accept_multiple_files=False, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
        with col2:
            img2= st.file_uploader ("Upload Template", type= ["jpg","png","bmp","jpeg"], accept_multiple_files=False, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
    
    
    
    

    with st.container():
        col1,col2 = st.columns([2,2])
        with col1:
           if img1:
                path1 = "images/" + img1.name
                image1 = cv2.imread(path1)
                image1 = cv2.resize(image1, (256, 256))
                st.image(path1, caption='Original Image' ,width=350,clamp=True) 
        with col2:
             if img2:
                path2 = "images/" + img2.name
                image2 = cv2.imread(path2)
                image2 = cv2.resize(image2, (256, 256))
                st.image(path2, caption='Template Image' ,width=350,clamp=True) 

        

    with st.container():
        col1,a,col2= st.columns([2,0.2,2]) 
        if (img1 and img2):
            t1 = time.time()
            keypoints1, descriptors1 = Sift.generateFeatures(image1)      
            keypoints2, descriptors2 = Sift.generateFeatures(image2) 
            rgbImg1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB) 
            imgplot1 = plt.imshow(rgbImg1)
            
            rgbImg2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB) 
            imgplot2 = plt.imshow(rgbImg2)
            

            for pnt in keypoints1:
                plt.scatter(pnt.pt[0], pnt.pt[1], s=pnt.size, c="red")
            for pnt in keypoints2:
                plt.scatter(pnt.pt[0], pnt.pt[1], s=pnt.size, c="red")
            t2 = time.time()
            print("Execution time of SIFT is {} sec".format(t2 - t1))         

    with st.container():

        col1,a,col2= st.columns([2,0.2,2]) 

        if (img1 and img2):
            with col1:
                start_time = time.time()
                ssd_score = MatchingFeatures.feature_matching(descriptors1,descriptors2,"SSD")
                end_time = time.time()
                match_time = end_time - start_time
                print("SSD computation time: ",match_time)
                matched_features = sorted(ssd_score, key=lambda x: x.distance, reverse=True)
                matched_image = cv2.drawMatches( img1=image1, keypoints1=keypoints1,img2= image2, keypoints2=keypoints2,matches1to2= matched_features[:30], outImg= image2, flags=2)
                st.image(matched_image, caption='SSD Image' ,width=350,clamp=False, output_format="auto") 
            
            with col2:
                tart_time = time.time()
                ncc_score = MatchingFeatures.feature_matching(descriptors1,descriptors2,"NCC")
                end_time = time.time()
                match_time = end_time - start_time
                print("NCC computation time: ",match_time)
                matched_features = sorted(ncc_score, key=lambda x: x.distance, reverse=True)
                matched_image = cv2.drawMatches( img1=image1, keypoints1=keypoints1,img2= image2, keypoints2=keypoints2,matches1to2= matched_features[:30], outImg= image2, flags=2)
                st.image(matched_image, caption='NCC Image' ,width=350,clamp=False, output_format="auto")
                        
            


#Thresholding tab
with tab8:

    with st.container():
        col1,a,col2,b,col3= st.columns([2,0.2,2,0.2,2]) 
        if main_image:
            path3 = "images/" + main_image.name
            image= cv2.imread(path3,cv2.IMREAD_GRAYSCALE) 
            with col1:
                st.image(image, caption='Original image' ,width=300) 
                    
            with col2:
                if main_image:
                # Global Option
                    imgread=cv2.imread(path3, 0)
                    image_array = np.array(imgread)
                    if(btn_threshold=='Optimal'):
                        # apply optimal global func
                        optimal_img= Optimal(image_array)
                        cv2.imwrite("images/optimal_global_output.png",optimal_img)
                        st.image("images/optimal_global_output.png", caption='Optimal Global' ,width=300,clamp=True) 
                    
                    if(btn_threshold=='Otsu'):
                        # Threshold the image using Otsu's thresholding
                        binary_image = otsu_threshold(image_array)
                        # Save the binary image
                        binary_image = Image.fromarray(np.uint8(binary_image * 255))
                        # binary_image.save('binary_image.png')
                        binary_image.show()
                        st.image(binary_image, caption='Otsu Global' ,width=300,clamp=True) 
                    
                    if(btn_threshold=='Spectral'):
                        # apply spectral global func
                        spectral_img = spectral(image_array)
                        cv2.imwrite("images/spectral_global_output.png",spectral_img)
                        st.image("images/spectral_global_output.png", caption='Spectral Global' ,width=300,clamp=True) 
                                  
            with col3: 
                if main_image:
                # local Option
                    if(btn_threshold=='Optimal'):
                        new_img=local_thresholding(image_array,4,Optimal )
                        cv2.imwrite("images/optimal_local_output.png",new_img)
                        st.image("images/optimal_local_output.png", caption='Optimal Local' ,width=300,clamp=True)
                    
                    if(btn_threshold=='Otsu'):
                        new_img=local_thresholding(image_array,4,otsu_threshold )
                        new_img=Image.fromarray(np.uint8(new_img * 255))
                        st.image(new_img, caption='Otsu Local' ,width=300,clamp=True) 
                    
                    if(btn_threshold=='Spectral'):
                        new_img=local_thresholding(image_array,4,spectral)
                        cv2.imwrite("images/spectral_local_output.png",new_img)
                        st.image("images/spectral_local_output.png", caption='Spectral Local' ,width=300,clamp=True) 
                       

          


#segmentation tab
with tab9:


    if main_image:
        path4 = "images/" + main_image.name
        rgb_img = cv2.imread(path4)
        b,g,r = cv2.split(rgb_img)       # get b,g,r
        img = cv2.merge([r,g,b])     # switch it to rgb
        cop_img = np.copy(img)

    with st.container():
        if main_image:
            # Convert the RGB image to LUV using your implemented function
            converted_image = rgb_to_luv(cop_img)
            # Display the original image and the segmented image
            fig, axs = plt.subplots(1, 2, figsize=(60, 60))
            axs[0].imshow(cop_img)
            axs[0].set_title('Original Image')
            axs[1].imshow(converted_image)
            axs[1].set_title('converted Image')
            plt.savefig("images/LUV_Output.png")
            st.image("images/LUV_Output.png", caption='LUV Converted image' ,width=600)


    with st.container():
        if main_image:
            if(btn_segment=='K-means'):
                segmented_image = kmeans_segmentation(img, k = cluster_number,max_iterations=iteration_number)

            
            if(btn_segment=='Region Growing'):
                cop_img = np.copy(rgb_img)
                segmented_image = apply_region_growing (img)
            
            if(btn_segment=='Agglomerative'):
                segmented_image = apply_agglomerative_clustering(img,clusters_number=final_cluster,initial_clusters_number=initial_cluster)
                
            
            if(btn_segment=='Mean Shift'):
                segmented_image = mean_shift(cop_img,60)


            # Display the original image and the segmented image
            fig, axs = plt.subplots(1, 2, figsize=(60, 60))
            axs[0].imshow(img)
            axs[0].set_title('Original Image')
            axs[1].imshow(segmented_image)
            axs[1].set_title('Segmented Image')
            plt.savefig("images/segmentedOutput.png")
            st.image("images/segmentedOutput.png", caption='Segmented image' ,width=600) 




with tab10:
    image_for_detect= st.file_uploader ("Upload Image ", type= ["jpg","png","bmp","jpeg","pgm"], accept_multiple_files=False, key='file1', help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
    
    with st.container():
        col1,col2 = st.columns([2,2])
        if image_for_detect:
            path_img = "images/" + image_for_detect.name
            imge=cv2.imread(image_for_detect,cv2.COLOR_BGR2RGB) 
            imge= cv2.imread(image_for_detect, cv2.IMREAD_GRAYSCALE)

        with col1:
            if(image_for_detect):
                st.image(image_for_detect, caption='Original image' ,width=300)

        with col2:
            if(image_for_detect):
                faces = detect_faces(path= path_img , scale_factor= scale_factor , min_neighbour= min_neighbour)
                for (x, y, w, h) in faces:
                    cv2.rectangle(imge, (x, y), (x+w, y+h), (255, 0, 0), 10)
                st.image(imge, caption='Detected face' ,width=300)
            




with tab11:
    image_recognize= st.file_uploader ("Upload Image ", type= ["jpg","png","bmp","jpeg","pgm"], accept_multiple_files=False, key='file2', help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
    
    with st.container():
        col1,col2 = st.columns([2,2])
        if image_recognize:
            path__Rec_img = image_recognize.name
            img_Rec=cv2.imread(image_recognize,cv2.COLOR_BGR2RGB) 
            img_Rec= cv2.imread(image_recognize, cv2.IMREAD_GRAYSCALE)

        with col1:
            if(image_recognize):
                st.image(image_recognize, caption='Original image' ,width=300)
        

        with col2:
            if(image_recognize):
                unknown_face = cv2.imread(path__Rec_img)#read the image
                gray = cv2.cvtColor(unknown_face, cv2.COLOR_BGR2GRAY)
                unknown_face = cv2.resize(gray, (width, height))#resize with 64*64 shape
                unknown_face = np.array(unknown_face, dtype='float64').flatten()
                train_labels, best_match = PCA_APPLY(unknown_face)

                st.image("images\FaceRecognized.png", caption='Best Match' ,width=300)
    

    with st.container():
        if(image_recognize):
            col1,a,col2,b,col3 = st.columns([2,0.2,2,0.2,2])

            st.subheader("Performance of the PCA Model")

            if(image_recognize):
                test_path="testing/"
                fpr_values,tpr_values,accuracy,precision,specificity = calculate_performance(test_path=test_path,threshold=Euclidean_threshold,width=width,height=height)


            with col1:
                st.markdown("Accuracy =  ")
                st.write(accuracy)
            
            with col2:
                st.markdown("Precision =  ")
                st.write(precision)
            
            with col3:
                st.markdown("Specificity =  ")
                st.write(specificity)
                    

        with st.container():
            if(image_recognize):

                ROC_plot(tpr_values=tpr_values,fpr_values=fpr_values)

                st.image("images/ROC_CURVE.png", caption='ROC Curve' ,width=300)