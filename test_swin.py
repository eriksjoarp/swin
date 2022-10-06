from torchvision.transforms import ToTensor
import gdal
import rasterio
from fastai.vision.data import ImageList
import PIL
import torch
import cv2

from ..helper import erik_functions_files
from ..ai_helper import dataset_load_helper


def openMultiChannelImage(fpArr):
    '''
    Open multiple images and return a single multi channel image
    '''
    mat = None
    nChannels = len(fpArr)
    for i, fp in enumerate(fpArr):
        img = PIL.Image.open(fp)
        chan = PIL.pil2tensor(img, dtype='float').div_(255)
        if (mat is None):
            mat = torch.zeros((nChannels, chan.shape[1], chan.shape[2]))
        mat[i, :, :] = chan
    return PIL.Image(mat)


#class MultiChannelImageItemList(ImageItemList):
class MultiChannelImageItemList(ImageList):
    def open(self, fn):
        return openMultiChannelImage(fn)



import cv2

path = r'''C:\\ai\datasets\\eurosat\\EuroSATallBands\\2750\\AnnualCrop\\AnnualCrop_1.tif'''

mylist = []
loaded, mylist = cv2.imreadmulti(mats = mylist, filename = path, flags = cv2.IMREAD_ANYCOLOR )

# alternative usage
#loaded,mylist = cv2.imreadmulti(mats = mylist, start =0, count = 2, filename = path, flags = cv2.IMREAD_ANYCOLOR )

cv2.imshow("mylist[0]",mylist[0])
cv2.imshow("mylist[1]",mylist[1])
cv2.waitKey()

exit(0)

path = r'''C:\ai\datasets\eurosat\EuroSATallBands\2750\AnnualCrop\AnnualCrop_1.tif'''

ret, images = cv2.imreadmulti(path)


exit(0)

#def get_filenames_in_dir(dir_src, extension ='NONE', full_path = False):
img_list = erik_functions_files.get_filenames_in_dir(r'C:\ai\datasets\eurosat\EuroSATallBands\2750\AnnualCrop', full_path=True)

img_list = img_list[:10]
print(len(img_list))

images = ImageList.from_folder(r'C:\ai\datasets\eurosat\EuroSATallBands\2750\AnnualCrop\AnnualCrop_1.tif')

img_tensors = dataset_load_helper.MultiChannelImageItemList(images)

print(img_tensors.shape)


exit(0)
image = rasterio.open(r'C:\ai\datasets\eurosat\EuroSATallBands\2750\AnnualCrop\AnnualCrop_1.tif')
image = image.read()
image = ToTensor()(image)

print(image.shape)


