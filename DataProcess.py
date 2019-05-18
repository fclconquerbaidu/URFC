import numpy as np
import glob
import os
import math
from scipy.stats import norm 
import matplotlib.pyplot as plt 
import cv2
import params
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import pandas
from imgaug import augmenters as iaa
from sklearn.model_selection import train_test_split
from albumentations import (
    PadIfNeeded,
    HorizontalFlip,
    VerticalFlip,    
    CenterCrop,    
    Crop,
    Compose,
    Transpose,
    RandomRotate90,
    ElasticTransform,
    GridDistortion, 
    OpticalDistortion,
    RandomSizedCrop,
    OneOf,
    CLAHE,
    RandomContrast,
    RandomGamma,
    RandomBrightness,
    RandomBrightnessContrast
)

def image_augmentation__(img):
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
    img_a=datagen.flow(img,batch_size=1)
    img_a=np.squeeze(img_a.x)
    img_a=np.array(img_a,np.uint8)
    return img_a
def image_augmentation(currImg):
    """
    Apply random image augmentations
    :param currImg: original image
    :param labelMask: original ground truth
    :return: post-augmentation image and ground truth data
    """
    aug = Compose([VerticalFlip(p=0.5), #RandomBrightnessContrast(p=0.05),             
            RandomRotate90(p=0.5),HorizontalFlip(p=0.5),Transpose(p=0.5)])

    augmented = aug(image=currImg)
    imageMedium = augmented['image']
    #imageMedium=np.squeeze(imageMedium)
    #visualize(imageMedium,labelMedium,currImg,labelMask)
    return imageMedium
def ImgNormalization(img):
    img_return=img/125.0-1
    return img_return
def VisitNormalization_xiao(feature):
    feature[:,0]=feature[:,0]/1400-1;
    feature[:,1:13]=feature[:,1:13]/0.085-1;
    feature[:,13]=feature[:,13]/3.57-1;
    feature[:,14]=feature[:,14]/0.3-1;
    feature[feature<-1]=-1;
    feature[feature>1]=1;
    return feature
def VisitNormalization(feature):
    for i in range(len(feature)):
        data=feature[i,:]
        max_num=max(data)

        data=(data-min(data))/max_num*8
        feature[i,:]=data
    return feature
def LoadVisitFeatures(txt_path):
    txt_name=os.path.split(txt_path)[-1]
    if len(txt_name)>11:
        txt_path=txt_path[:-8]+'.txt'
    if not os.path.exists(txt_path):
        print('no such visit txt file: ',txt_path)
    fp = open(txt_path)
    lines = fp.readlines()
    fp.close()
    featur=[]
    for line in lines:
        line = line.strip('\n')
        numbers=line.split(',')
        for num in numbers:
            if num!='':
                featur.append(float(num))
                
    return featur
def input_generator_p(img_files, visit_file, batch_size):
    if len(img_files)!= len(visit_file) or len(img_files)!= len(lable):
        print('the input data have not equal length!!!')    
    N = len(img_files) #total number of images

    idx = np.random.permutation(N) #shuffle the order

    batchInds = get_batch_inds(batch_size, idx, N) # generate each batch's idx like this:{[10,23,....],[1,1232,44,...],[....]....} 

    #executor = ProcessPoolExecutor(max_workers=3)

    while True:
        for inds in batchInds:
            img_batch = [img_files[ind] for ind in inds]
            label_batch = [lable[ind] for ind in inds]
            visit_batch=[visit_file[ind] for ind in inds]
            imgdata, visit, gts= load_cnn_batch(img_batch, label_batch,visit_file)#, executor)      
            
            if 0:
                import matplotlib.pyplot as plt 
                plt.subplot(221) #用于显示多个子图121代表行、列、位置
                plt.imshow(imgdata[0,:,:,:])
                plt.title('org')
                plt.subplot(222)
                plt.imshow(imgdata[1,:,:,:])
                plt.title('1') #添加标题
                plt.subplot(223)
                plt.imshow(imgdata[2,:,:,:])
                plt.title('2') #添加标题
                plt.subplot(224)
                plt.imshow(imgdata[3,:,:,:])
                plt.title('3') #添加标题
                plt.show()
            #if visit_file>0
            if params.use_metadata:
                yield ([imgdata, visit], gts)
                #return ([imgdata, visit], gts)
            else:
                yield (imgdata, gts)
                #return (imgdata,gts)

def get_batch_inds(batch_size, idx, N,predict=False):
    """
    Generates an array of indices of length N
    :param batch_size: the size of training batches
    :param idx: data to split into batches
    :param N: Maximum size
    :return batchInds: list of arrays of data of length batch_size
    """
    batchInds = []
    idx0 = 0

    toProcess = True
    while toProcess:
        idx1 = idx0 + batch_size
        if idx1 >= N:
            idx1 = N
            if predict==False:

                idx0 = idx1 - batch_size
            toProcess = False
        batchInds.append(idx[idx0:idx1])
        idx0 = idx1

    return batchInds

def input_generator(img_files, lable,visit_file, batch_size):
    if len(img_files)!= len(visit_file) or len(img_files)!= len(lable):
        print('the input data have not equal length!!!')    
    N = len(img_files) #total number of images

    idx = np.random.permutation(N) #shuffle the order

    batchInds = get_batch_inds(batch_size, idx, N) # generate each batch's idx like this:{[10,23,....],[1,1232,44,...],[....]....} 

    #executor = ProcessPoolExecutor(max_workers=3)

    while True:
        for inds in batchInds:
            img_batch = [img_files[ind] for ind in inds]
            label_batch = [lable[ind] for ind in inds]
            visit_batch=[visit_file[ind] for ind in inds]
            imgdata, visit, gts= load_cnn_batch(img_batch, label_batch,visit_file)#, executor)      
            
            if 0:
                import matplotlib.pyplot as plt 
                plt.subplot(221) #用于显示多个子图121代表行、列、位置
                plt.imshow(imgdata[0,:,:,:])
                plt.title('org')
                plt.subplot(222)
                plt.imshow(imgdata[1,:,:,:])
                plt.title('1') #添加标题
                plt.subplot(223)
                plt.imshow(imgdata[2,:,:,:])
                plt.title('2') #添加标题
                plt.subplot(224)
                plt.imshow(imgdata[3,:,:,:])
                plt.title('3') #添加标题
                plt.show()
            #if visit_file>0
            if params.use_metadata:
                yield ([imgdata, visit], gts)
                #return ([imgdata, visit], gts)
            else:
                yield (imgdata, gts)
                #return (imgdata,gts)

def load_cnn_batch(img_batch, label_batch,visit_batch):
    """
    """
    results=[]
    imgdata=[]
    labels=[]
    visits=[]
    futures = []

    for i in range(0, len(img_batch)):
        currInput = {}
        currInput['gts'] =label_batch[i]
        currInput['rgb'] = img_batch[i]
        currInput['visit'] = visit_batch[i]
        #futures.append(executor.submit(_load_batch_helper, currInput))
        results.append(_load_batch_helper(currInput))

        

    #results = [future.result() for future in futures]
 
    for  i, result in enumerate(results):
        imgdata.append(result[0][0])
        visits.append(result[1][0])
        labels.append(result[2][0])
        if params.data_augmentation:
            imgdata.append(result[0][1])
            visits.append(result[1][1])
            labels.append(result[2][1])

    y_train=np.array(labels, np.float32)
    x_train = np.array(imgdata, np.float32)
    visits= np.array(visits, np.float32)
            
    y_train = to_categorical(y_train, params.num_labels)
    x_train=ImgNormalization(x_train)
    visits=VisitNormalization(visits)



    return x_train, visits,y_train

def _load_batch_helper(inputDict):
    """
    Helper for load_cnn_batch that actually loads imagery and supports parallel processing
    :param inputDict: dict containing the data and metadataStats that will be used to load imagery
    :return currOutput: dict with image data, metadata, and the associated label
    """
    #print("fsf")
    #return

    rgb_file = inputDict['rgb']
    gts_file = inputDict['gts']
    visit_file=inputDict['visit']
    inputs=[]
    labels=[]
    visits=[]
    img_data=cv2.imread(rgb_file)
    inputs.append(img_data)
    labels.append(gts_file)    
    if len(visit_file)>0:
        visits.append(LoadVisitFeatures(visit_file))
   # img_data_ = img_data.reshape((1,) + img_data.shape)
    if params.data_augmentation:
        imageMedium = image_augmentation(img_data)
        inputs.append(imageMedium)
        labels.append(gts_file)    
        if len(visit_file)>0:
            visits.append(LoadVisitFeatures(visit_file))


    if 0:
                import matplotlib.pyplot as plt 
                plt.subplot(121) #用于显示多个子图121代表行、列、位置
                plt.imshow(img_data)
                plt.title('org')
                plt.subplot(122)
                plt.imshow(imageMedium)
                plt.title('translated') #添加标题
                plt.show()
    return inputs, visits, labels


def load_all_data_test(data_folder):
    imgs=[]
    #img_folder=os.path.join(data_folder,test_img_folder)
    img_files = glob.glob(os.path.join(data_folder, '*/*.jpg'))

    for imgPath in img_files:
        imageName = os.path.split(imgPath)[-1]
        imgs.append(imgPath)
    ###########################
    ortho_list_file=os.path.join(data_folder,'test_img_list.txt')

    f_ortho = open(ortho_list_file,'w')
 
    for i in range(len(imgs)):
        f_ortho.write(imgs[i]+'\n');
    f_ortho.close()
    return imgs
def GetPredictData(img_path,visit_path):
    inputs=[]
    visits=[]
    img_data=cv2.imread(img_path)
    inputs.append(img_data)
    visits.append(LoadVisitFeatures(visit_path))
    for i in range (0):
        imageMedium = image_augmentation(img_data)
        inputs.append(imageMedium)
        visits.append(LoadVisitFeatures(visit_path))
    inputs=np.array(inputs,np.float32)
    visits=np.array(visits)
    inputs=ImgNormalization(inputs)
    visits=VisitNormalization(visits)
    return inputs, visits    

def GetFinalPredbyVotes(preds):
    class_vote=[0]*params.num_labels
    for pred in preds:
        label=pred
        class_vote[label]=class_vote[label]+1
    best_label=np.argmax(class_vote, axis=-1).astype('uint8')
    return best_label

def LoadAllVisitFeatures(file_path,output_folder):
    if os.path.exists(output_folder)==0:
        os.mkdir(output_folder)  
    data=pandas.read_pickle(file_path)
    data=np.array(data)
    for i in range (len(data)):
        values=data[i,:]
        txt_nam=values[0]+'.txt'
        out_path=os.path.join(output_folder,txt_nam)
        values=values[1:-1]
        f=open(out_path,'w')
        for value in values:
            f.write(str(value)+',')
        f.close()
        
    return data


def augumentor(image):
    augment_img = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.SomeOf((0,4),[
            iaa.Affine(rotate=90),
            iaa.Affine(rotate=180),
            iaa.Affine(rotate=270),
            iaa.Affine(shear=(-16, 16)),
        ]),
        iaa.OneOf([
                iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
            ]),
        #iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
        ], random_order=True)

    image_aug = augment_img.augment_image(image)
    return image_aug

def LoadAllTrainData(img_folder, visit_folder):
    imgs_t=[]
    visits_t=[]
    labels_t=[]
    imgs_v=[]
    visits_v=[]
    labels_v=[]
    #img_folder=os.path.join(data_folder,test_img_folder)
    img_files = glob.glob(os.path.join(img_folder, '*.jpg'))

    train_data_list,val_data_list = train_test_split(img_files, test_size=0.1, random_state = 2050)
    for imgPath in train_data_list:
        img=cv2.imread(imgPath)
        imageName = os.path.split(imgPath)[-1]
        visitfile=imageName[:-3]+'npy'
        visitPath=os.path.join(visit_folder,visitfile)
        visit=np.load(visitPath)
        visit=visit.transpose()
        label=imageName[-5:-4]
        labels_t.append(int(label)-1)
        imgs_t.append(img)
        visits_t.append(visit)
    for imgPath in val_data_list:
        img=cv2.imread(imgPath)
        imageName = os.path.split(imgPath)[-1]
        visitfile=imageName[:-3]+'npy'
        visitPath=os.path.join(visit_folder,visitfile)
        visit=np.load(visitPath)
        visit=visit.transpose()
        imgs_v.append(img)
        visits_v.append(visit)
        label=imageName[-5:-4]
        labels_v.append(int(label)-1)
    ###########################
    return imgs_t,visits_t,labels_t,imgs_v,visits_v,labels_v
def InputGenerator_data(img_files,visit_file,lable,batch_size):
    if len(img_files)!= len(visit_file) or len(img_files)!= len(lable):
        print('the input data have not equal length!!!')    
    N = len(img_files) #total number of images

    idx = np.random.permutation(N) #shuffle the order

    batchInds = get_batch_inds(batch_size, idx, N) # generate each batch's idx like this:{[10,23,....],[1,1232,44,...],[....]....} 

    #executor = ProcessPoolExecutor(max_workers=3)


    while True:
        for inds in batchInds:
            img_batch=[]
            for ind in inds:
                img=img_files[ind]
                img=augumentor(img)
                img_batch.append(img)
            #img_batch = [img_files[ind] for ind in inds]
            label_batch = [lable[i] for i in inds]
            visit_batch=[visit_file[i] for i in inds]
            img_batch=np.array(img_batch).astype(float)
            img_batch=(img_batch/125-1.0)
            visit_batch=np.array(visit_batch).astype(float)
            label_batch=np.array(label_batch)
            label_batch = to_categorical(label_batch, params.num_labels)
            if 0:
                import matplotlib.pyplot as plt 
                plt.subplot(221) #用于显示多个子图121代表行、列、位置
                plt.imshow(imgdata[0,:,:,:])
                plt.title('org')
                plt.subplot(222)
                plt.imshow(imgdata[1,:,:,:])
                plt.title('1') #添加标题
                plt.subplot(223)
                plt.imshow(imgdata[2,:,:,:])
                plt.title('2') #添加标题
                plt.subplot(224)
                plt.imshow(imgdata[3,:,:,:])
                plt.title('3') #添加标题
                plt.show()
            #if visit_file>0
            if params.use_metadata:
                yield ([img_batch, visit_batch], label_batch)
                #return ([img_batch, visit_batch], label_batch)
            else:
                yield (img_batch, label_batch)
                #return (imgdata,gts)    

if __name__ == "__main__":
    # data_file='G:/DataSet/UrbanClassification/data/training_data_7_24.pkl'
    # output_folder='G:/DataSet/UrbanClassification/data/train_visit_feature_he'
    # LoadAllVisitFeatures(data_file,output_folder)
    img_folder=r'G:\DataSet\UrbanClassification\data\train'
    visit_folder=r'G:\DataSet\UrbanClassification\data\npy\train_visit'
    LoadAllTrainData(img_folder,visit_folder)
