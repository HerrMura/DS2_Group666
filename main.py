# -*- coding: utf-8 -*-
import cv2
import os
import streamlit as st
from PIL import Image
from PIL import ImageDraw
from SR_main import get_SR_result
from SR_main import get_Video_result

#说明
def instruction():
    """ 运行方式
        
        安装streamlit并运行该py文件后，在该目录的cmd中运行以下命令
            streamlit run 该文件.py
        会自动弹出网页前端，网页运行时须保持cmd一直运行
    
    """

#拍照并得到图片
def take_photo():
    """ 拍照并得到图片，图片自动保存为Photo.jpg
    
        输出:
            bytes_data: 图片（<class 'bytes'>）
            img: 图片（PIL Image）
            img_array： 图片（img_array = np.array(img)）
            st.download_button： 下载图片按钮
    """
    picture = st.camera_input("Take a picture")
    
    if picture:
        
        st.image(picture)
        # To read image file buffer as bytes:
        bytes_data = picture.getvalue()
        # Check the type of bytes_data:
        #st.write(type(bytes_data))# Should output: <class 'bytes'>
        
        # To read image file buffer as a PIL Image:
        img = Image.open(picture)
        # To convert PIL Image to numpy array:
        #img_array = np.array(img)
        # Check the type of img_array:
        #st.write(type(img_array))# Should output: <class 'numpy.ndarray'>
        # Check the shape of img_array:
        #st.write(img_array.shape)# Should output shape: (height, width, channels)
        
        st.download_button(label="Download Photo", data=picture, file_name="Photo.jpg", mime="image/jpg")
        # Save the Photo
        img.save("Photo.jpg")
     
#展示结果
def show_results(result_name):
    if os.path.exists(result_name):
        st.image(result_name, caption = result_name)
    else:
        st.write("Image does not exist!")
    
#展示图片细节（矩形）
def show_detail(name, box, selected_size):
    """ 显示图片细节

        输入:
            img: 待展示的图片（PIL Image）
            box: (矩形左上角点的横坐标，矩形左上角点的纵坐标，矩形右下角点的横坐标，矩形右下角的纵坐标)，例如(100, 100, 200, 200)

        输出:
            img: 图片带选定的红色矩形
            img_z: 图片的红色矩形部分放大后的图片
    """
    img = Image.open(name)
    a = ImageDraw.ImageDraw(img)
    #显示图片中的红色矩形
    a.rectangle(box, fill = None, outline = 'red', width = 4)
    #img.save()
    with st.expander(name, expanded = False):#expanded = False 默认不展开
        st.image(img, caption = name)
    
    #将图片中的红色矩形部分放大
    img_z = img.resize(selected_size, resample = Image.Resampling.LANCZOS, box = box)
    #img_z.save()
    st.image(img_z, caption = "detail zoomed in " + name)
    
#图片放大局部
def zoom_detail(result_name, point_upper_left_x, point_upper_left_y, point_lower_right_x, point_lower_right_y):
    """ 选择矩形大小，并获得局部放大的图片

        point_upper_left_x: 矩形左上角点的横坐标
        point_upper_left_y: 矩形左上角点的纵坐标
        point_lower_right_x： 矩形右下角点的横坐标
        point_lower_right_y： 矩形右下角的纵坐标
        
    """
    
    box = (point_upper_left_x, point_upper_left_y, point_lower_right_x, point_lower_right_y)
    selected_size = (point_lower_right_x - point_upper_left_x, point_lower_right_y - point_upper_left_y)
    st.header("Zoomed area")
    st.write(selected_size)
    try:
        if os.path.exists(result_name['SRCNN']):
            show_detail(result_name['SRCNN'], box, selected_size)
    except:
        pass
    try:
        if os.path.exists(result_name['SRPO']):
            show_detail(result_name['SRPO'], box, selected_size)
    except:
        pass
    try:
        if os.path.exists(result_name['Bicubic']):
            show_detail(result_name['Bicubic'], box, selected_size)
    except:
        pass

#缩放图片细节
def zoom_img(result_name):
    
    a = []
    b = []
    
    if os.path.exists(result_name['SRCNN']):
        img = cv2.imread(result_name['SRCNN'])
    elif os.path.exists(result_name['SRPO']):
        img = cv2.imread(result_name['SRPO'])
    elif os.path.exists(result_name['Bicubic']):
        img = cv2.imread(result_name['Bicubic'])
    else:
        st.write("Results do not exist!")
        return
    
    def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            xy = "%d, %d" % (x, y)
            a.append(x)
            b.append(y)
            cv2.circle(img, (x ,y), 1, (255, 0, 0), thickness = -1)
            cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 1)
            cv2.imshow("image", img)
            
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
    cv2.imshow("image", img)
    cv2.waitKey(0)


    if len(a)>1 and len(b)>1:
        if a[0]>a[1]:
            a[0], a[1] = a[1], a[0]
        if b[0]>b[1]:
            b[0], b[1] = b[1], b[0]
        point_upper_left_x = a[0]
        point_upper_left_y = b[0]
        point_lower_right_x = a[1]
        point_lower_right_y = b[1]
            
        zoom_detail(result_name, point_upper_left_x, point_upper_left_y, point_lower_right_x, point_lower_right_y)
    else:
        st.write("Input Error!")

    cv2.destroyAllWindows()


#加入侧边栏
checkbox_image_super_resolution = st.sidebar.checkbox('Image')
checkbox_video_super_reslolution = st.sidebar.checkbox('Video')


if checkbox_image_super_resolution:
    
    picture = None
    image_result_name = {}
    
    #----------------网页标题----------------
    st.title('Image Super-Resolution'.center(33))
    
    
    #----------------拍照功能----------------
    #Take Photo
    st.sidebar.subheader('Take Photo')
    checkbox_take_photo = st.sidebar.checkbox('Start taking a picture')
    if checkbox_take_photo:
        st.header('Take Photo')
        take_photo()
    
    #---------利用训练后的模型进行测试------------
    st.header('Test')
    #选择模型按钮
    model_options = st.multiselect('Model Selection', ['SRCNN', 'SRPO', 'Bicubic'])
    #选择待测试图片按钮
    uploaded_image = st.file_uploader("Choose an Image")
    #Denoise参数
    denoise_image = st.checkbox('Denoise')
    #选择缩放下拉栏
    image_option_scale = st.selectbox('Scale', (2, 3, 4))
    if model_options is not None and uploaded_image is not None and image_option_scale is not None:
        image_result_name = {
            'SRCNN' : uploaded_image.name.replace('.', '_SRCNN_x{}.'.format(image_option_scale)), 
            'SRPO' : uploaded_image.name.replace('.', '_SRPO_x{}.'.format(image_option_scale)), 
            'Bicubic' : uploaded_image.name.replace('.', '_Bicubic_x{}.'.format(image_option_scale))}
    
    if model_options is not None and uploaded_image is not None:
        if st.button('Get result'):
            st.header('Results')
            st.image(uploaded_image, caption = "Origin")
            with st.spinner('Wait for it...'):
                for image_option_model in model_options:
                    SR_result = get_SR_result(uploaded_image, image_option_scale, image_option_model, denoise_image)
                    #--------------测试结果展示---------------
                    show_results(image_result_name[image_option_model])
            st.success('Done!')
            
            
    else:
        st.caption('Test did not start!')
        
    #----------图片局部放大功能(侧边栏)-----------
    #Zoom Image
    st.sidebar.subheader('Zoom Detail')
    checkbox_zoom_detail = st.sidebar.checkbox('Zoom an area')
    if checkbox_zoom_detail and image_result_name:
        zoom_img(image_result_name)


if checkbox_video_super_reslolution:
    
    #----------------网页标题----------------
    st.title('Video Super-Resolution'.center(33))
    
    #---------利用训练后的模型进行测试------------
    st.header('Test')
    #选择模型按钮
    video_option_model = "SRPO"
    #选择待测试图片按钮
    uploaded_video = st.file_uploader("Choose a Video")
    #Denoise参数
    denoise_video = st.checkbox('Denoise')
    #选择缩放下拉栏
    video_option_scale = st.selectbox('Scale ', (2, 3, 4))
    if uploaded_video is not None and video_option_scale is not None:
        if st.button('Get result'):
            st.header('Results')
            st.caption("Origin")
            video_origin = open(uploaded_video.name, 'rb')
            video_origin_bytes = video_origin.read()
            st.video(video_origin_bytes)
            
            with st.spinner('Wait for it...'):
                Vid_result = get_Video_result(uploaded_video.name, video_option_scale, denoise_video)
                st.caption("Result")
                if denoise_video==True: 
                    video_result = open('./Output/Final_denoised.mp4', 'rb')
                else:
                    video_result = open('./Output/Final.mp4', 'rb')
                video_result_bytes = video_result.read()
                st.video(video_result_bytes)
            st.success('Done!')
