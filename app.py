import streamlit as st
import cv2
import Hough , ActiveContour
from filter import *
from histogram import *
from frequency import *
import numpy as np
import seaborn as sns
from skimage import io
from skimage.filters import gaussian
import matplotlib.pyplot as plt
from skimage.feature import canny
from skimage.draw import ellipse_perimeter






#page styling
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(page_title='Shape Detections', layout="wide")

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



tab1, tab2,tab3,tab4,tab5 = st.tabs(["Filters", "Histogram", "Frequency","Shape Detections", "Contour"])


with st.sidebar:
    main_image= st.file_uploader ("Upload Image ", type= ["jpg","png","bmp","jpeg"], accept_multiple_files=False, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")


#filters tab
with tab1:

    with st.sidebar:
        with st.expander("Filter and Edge Detection", expanded=False):
            btn3= st.selectbox('Edge detection ',('Sobel', 'Roberts', 'Prewitt' ,'Canny'))
            btn1= st.selectbox('Add noise to image',('Uniform', 'Gaussian', 'salt & pepper'))
            btn2= st.selectbox('Filter Image',('Average', 'Gaussian', 'median'))
            btn_filter=st.selectbox("choose size",('3','5'))


       
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

    with st.sidebar:
        with st.expander("Histogram",expanded=False):
            btn3=st.selectbox("Thresholding",options=('local','global'))
            btn4=st.selectbox("Processing",options=('Equalization','Normalization')) 

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
                if(btn3=='local'):
                    Local=local_thresholding(im2,60,60,60,60)
                    st.image(Local, caption='Local Threshold image' ,width=300) 
                elif(btn3=='global'):
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

   
    with st.sidebar:

        with st.expander("Detection Options",expanded=False):
            btn3= st.selectbox(' Detection Type',('Lines', 'Circles' ,'Ellipses'))

            if main_image:
                lower_threshold= st.slider("Lower Edge Threshold",min_value= 0,max_value= 255,value= 100 ,step=1)
                upper_threshold= st.slider("Upper Edge Threshold",min_value= 0,max_value= 255,value= 200 ,step=1)

                if(btn3=='Lines' or btn3 == 'Circles'):
                    threshold= st.slider("Threshold",min_value= 0,max_value= 400,value= 150 ,step=1)
            
                

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
                
                if(btn3=='Lines'):
                    Hough.Hough_line_detection(path=path,l_threshold=lower_threshold,u_threshold=upper_threshold,threshold=threshold)
                    st.image("images/Detected_Line_Image.jpg", caption='Line detected image' ,width=300,clamp=True) 



                elif(btn3=='Circles'):
                    Hough.Hough_circle_detection(path=path,threshold=threshold,l_threshold=lower_threshold,u_threshold=upper_threshold)
                    st.image("images\Circle_Detected_Image.jpg", caption='circle detected image' ,width=300,clamp=True) 
                
                
                elif(btn3=='Ellipses'):
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
    
    
    with st.sidebar:
        with st.expander("Active Contour",expanded=False):
             # user inputs for active contour paramaters
            alpha = st.number_input('Alpha',min_value=0.0, max_value=10.0, value=0.1, step=0.1)
            beta = st.number_input('Beta',min_value=0.0, max_value=10.0, value=0.1, step=0.1)
            gamma = st.number_input('Gamma',min_value=0.0, max_value=10.0, value=0.1, step=0.1)
            w_line = st.number_input('w-line',min_value=-10, max_value=10, value=0, step=1)
            w_edge = st.number_input('w-edge',min_value=-10, max_value=10, value=1, step=1)
   
   

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