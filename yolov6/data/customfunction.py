import cv2
from  skimage import exposure
import numpy as np
from matplotlib.gridspec import GridSpec
from PIL import Image

def fourth_binary_channel(im):

    im=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    cv2.imwrite("img3.jpg",im)
    im=exposure.equalize_hist(exposure.adjust_gamma(exposure.adjust_log(exposure.adjust_sigmoid(im))))
    cv2.imwrite("gray33.jpg",np.uint8(im*255))
    se=cv2.getStructuringElement(cv2.MORPH_RECT , (8,8))
    bg=cv2.morphologyEx(im, cv2.MORPH_DILATE, se)
    out_gray=cv2.divide(im, bg, scale=255)
    cv2.imwrite("gray4.jpg",out_gray)
    out_binary=cv2.threshold(np.uint8(out_gray), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1] 
    
    
    cv2.imwrite("binary3.jpg",out_binary)
    return im,out_binary

im=cv2.imread("/home/zeineddine/ICIPCompet/zeineddine-142224/datanew/dataset/images/val/000000297898.jpg")
fourth_binary_channel(im)


# import matplotlib.pyplot as plt
# import glob
# import random
# paths=glob.glob("/home/zeineddine/ICIPCompet/zeineddine-142224/datanew/dataset/images/val/*.jpg")


# def setBold(txt): return r"$\bf{" + str(txt) + "}$"
# for j in range(20):
#     plt.figure(figsize=(10,15))
#     for i in range(6):
#         n=random.randint(0,len(paths))
#         plt.subplot(2,3,i+1)
#         plt.axis('off')
#         img=cv2.imread(paths[n])
#         img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#         img=cv2.resize(img,(640,640),)
#         plt.imshow(img)
        
#     plt.subplots_adjust(wspace=0.05, hspace=-0.7)
#     plt.margins(0,0)
#     plt.savefig("./figures/fig"+str(n)+".jpg",transparent = True, bbox_inches = 'tight', pad_inches = 0)
#     #plt.show()

#     #plt.tight_layout()
#     print("aaaaa")

# fig=plt.figure(figsize=(12,15))
# gs=GridSpec(3,5,figure=fig)

# ax1=fig.add_subplot(gs[:,-1])
# ax2=fig.add_subplot(gs[0,0])
# ax3=fig.add_subplot(gs[0,1])
# ax4=fig.add_subplot(gs[0,2])
# ax5=fig.add_subplot(gs[0,3])
# ax6=fig.add_subplot(gs[1,0])
# ax7=fig.add_subplot(gs[1,1])
# ax8=fig.add_subplot(gs[1,2])
# ax9=fig.add_subplot(gs[1,3])
# ax10=fig.add_subplot(gs[2,0])
# ax11=fig.add_subplot(gs[2,1])
# ax12=fig.add_subplot(gs[2,2])
# ax13=fig.add_subplot(gs[2,3])


# scale=Image.open("./yolov6/data/scale.png")
# scale=scale.resize((scale.size[0],int(scale.size[1]*0.8)),Image.ANTIALIAS)
# ax1.imshow(scale)
# ax1.axis('off')
# ax1.margins(0,0)
# img=Image.open("./yolov6/data/img.jpg").resize((640,640),Image.ANTIALIAS)

# ax2.imshow(img,cmap='gray')
# ax2.axis('off')
# ax2.margins(0,0)
# ax2.set_title("Images from DCCOCO dataset in grayscale",fontsize=17 )
# img_quality=Image.open("./yolov6/data/img1_quality.png").resize((640,640),Image.ANTIALIAS)
# ax3.imshow(img_quality)
# ax3.axis('off')
# ax3.margins(0,0)
# ax3.text(-40,700,"Good quality %:67.14 - Degradation %:29.57", fontsize=18, ha='center', va='center')
# ax3.set_title("Quality assessment for grayscale images",fontsize=17 )
# gray=Image.open("./yolov6/data/gray.jpg").resize((640,640),Image.ANTIALIAS)
# ax4.imshow(gray,cmap='gray')
# ax4.axis('off')
# ax4.margins(0,0)
# ax4.set_title("Enhanced grayscale images",fontsize=17 )
# gray_quality=Image.open("./yolov6/data/gray1_quality.png").resize((640,640),Image.ANTIALIAS)
# ax5.imshow(gray_quality)
# ax5.axis('off')
# ax5.text(-40,700,"Good quality %:70.12 - Degradation %:26.90", fontsize=18, ha='center', va='center')
# ax5.margins(0,0)
# ax5.set_title("Quality assessment for enhanced grayscale images",fontsize=17 )
# img2=Image.open("./yolov6/data/img2.jpg").resize((640,640),Image.ANTIALIAS)
# ax6.imshow(img2,cmap='gray')
# ax6.axis('off')
# ax7.margins(0,0)
# img2_quality=Image.open("./yolov6/data/img2_quality.png").resize((640,640),Image.ANTIALIAS)
# ax7.imshow(img2_quality)
# ax7.axis('off')
# ax7.text(-40,700,"Good quality %:48.84 - Degradation %:46.04", fontsize=18, ha='center', va='center')

# ax7.margins(0,0)
# gray2=Image.open("./yolov6/data/gray2.jpg").resize((640,640),Image.ANTIALIAS)
# ax8.imshow(gray2,cmap='gray')
# ax8.axis('off')
# ax8.margins(0,0)
# gray2_quality=Image.open("./yolov6/data/gray2_quality.png").resize((640,640),Image.ANTIALIAS)
# ax9.imshow(gray2_quality)
# ax9.axis('off')
# ax9.margins(0,0)
# ax9.text(-40,700,"Good quality %:65.13 - Degradation %:31.39", fontsize=18, ha='center', va='center')

# img3=Image.open("./yolov6/data/img3.jpg").resize((640,640),Image.ANTIALIAS)
# ax10.imshow(img3,cmap='gray')
# ax10.axis('off')
# ax10.margins(0,0)
# img3_quality=Image.open("./yolov6/data/img3_quality.png").resize((640,640),Image.ANTIALIAS)
# ax11.imshow(img3_quality)
# ax11.axis('off')
# ax11.margins(0,0)
# ax11.text(-40,700,"Good quality %:70.79 - Degradation %:26.29", fontsize=18, ha='center', va='center')

# gray3=Image.open("./yolov6/data/gray3.jpg").resize((640,640),Image.ANTIALIAS)
# ax12.imshow(gray3,cmap='gray')
# ax12.axis('off')
# ax12.margins(0,0)
# gray3_quality=Image.open("./yolov6/data/gray3_quality.png").resize((640,640),Image.ANTIALIAS)
# ax13.imshow(gray3_quality)
# ax13.axis('off')
# ax13.margins(0,0)
# ax13.text(-40,700,"Good quality %:73.43 - Degradation %:23.91", fontsize=18, ha='center', va='center')

# #plt.subplots_adjust(wspace=0.05, hspace=-0.7)
# gs.tight_layout(fig)
# gs.update(wspace=-0.2, hspace=0.2,bottom=0.05)

# #plt.savefig("./quality.jpg",transparent = True, bbox_inches = 'tight', pad_inches = 0)
# plt.show()
