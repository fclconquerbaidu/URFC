
import json
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.applications import imagenet_utils
from data_ml_functions.mlFunctions import get_cnn_model,img_metadata_generator,get_lstm_model,codes_metadata_generator
from data_ml_functions.dataFunctions import prepare_data,calculate_class_weights
import numpy as np
import os
import tensorflow as tf
from keras import backend as K
#from data_ml_functions.multi_gpu import make_parallel
from keras.callbacks import ModelCheckpoint,CSVLogger
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import params
import time

from DataProcess import input_generator,load_all_data_test,GetPredictData,GetFinalPredbyVotes,LoadAllTrainData,InputGenerator_data
from VisitData import load_class_balanced_files,input_load_train_data
class URFCBaseline:
    def __init__(self, params=None, argv=None):
        """
        Initialize baseline class, prepare data, and calculate class weights.
        :param params: global parameters, used to find location of the dataset and json file
        :return: 
        """
        self.params=params
        # TensorFlow allocates all GPU memory up front by default, so turn that off
        config = tf.ConfigProto()
        os.environ["CUDA_VISIBLE_DEVICES"]='3'
        config.gpu_options.allow_growth = True
        K.tensorflow_backend.set_session(tf.Session(config=config))
    def train_cnn_old(self, img_folder,check_folder=params.directories['cnn_checkpoint_weights'],pre_trained_weight_path=''):
        """
        Train CNN with or without metadata depending on setting of 'use_metadata' in params.py.
        :param: 
        :return: 
        """
        if os.path.exists(check_folder)==0:
            os.mkdir(check_folder)        
        img_file_t,  lable_t,visit_file_t, img_file_v, label_v, visit_file_v=load_class_balanced_files(img_folder,max_samples=-100)
        
        nb_epoch=500
        batch_size=round(128)
        n_batch_per_epoch=len(img_file_t)//batch_size

        num_val_sample=len(img_file_v)
        n_batch_per_epoch_val=num_val_sample//batch_size
        # data_train, label_train=input_load_train_data(img_file_t,visit_file_t, lable_t)
        # data_val, label_val=input_load_train_data(img_file_v,visit_file_v, label_v)
        train_generator=input_generator(img_file_t, lable_t,visit_file_t, batch_size)
        val_generator = input_generator(img_file_v,label_v,visit_file_v,batch_size)

        model = get_cnn_model(self.params)
        #model = make_parallel(model, 4)
        if len(pre_trained_weight_path)>1:
            model.load_weights(pre_trained_weight_path,True)
            print('use pre-trained weights: ',pre_trained_weight_path)            
        model.compile(optimizer=Adam(lr=self.params.cnn_adam_learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

        print("training")
        filePath = os.path.join(check_folder, 'weights.{epoch:02d}.hdf5')
        checkpoint = ModelCheckpoint(filepath=filePath, monitor='loss', verbose=0, save_best_only=False,
                                     save_weights_only=False, mode='auto', period=5)
        csv_logger = CSVLogger(os.path.join(check_folder,'train.csv'))
        callbacks_list = [csv_logger,checkpoint]

        model.fit_generator(train_generator, steps_per_epoch=n_batch_per_epoch,
                            validation_data=val_generator, validation_steps=n_batch_per_epoch_val,
                            epochs=nb_epoch, callbacks=callbacks_list)

        #model.fit(data_train,label_train, batch_size=batch_size,validation_data=(data_val, label_val),
        #                     epochs=nb_epoch, callbacks=callbacks_list)
 

    def train_cnn(self, img_folder,visit_folder, check_folder=params.directories['cnn_checkpoint_weights'],pre_trained_weight_path=''):
        """
        Train CNN with or without metadata depending on setting of 'use_metadata' in params.py.
        :param: 
        :return: 
        """
        if os.path.exists(check_folder)==0:
            os.mkdir(check_folder)        
        img_t,  visit_t,label_t, img_v,  visit_v,label_v=LoadAllTrainData(img_folder,visit_folder)
        
        nb_epoch=500
        batch_size=round(128)
        n_batch_per_epoch=len(img_t)//batch_size

        num_val_sample=len(img_v)
        n_batch_per_epoch_val=num_val_sample//batch_size
        # data_train, label_train=input_load_train_data(img_file_t,visit_file_t, lable_t)
        # data_val, label_val=input_load_train_data(img_file_v,visit_file_v, label_v)
        train_generator=InputGenerator_data(img_t, visit_t, label_t,batch_size)
        val_generator = InputGenerator_data(img_v, visit_v, label_v,batch_size)
                        #input_generator
        model = get_cnn_model(self.params)
        #model = make_parallel(model, 4)
        if len(pre_trained_weight_path)>1:
            model.load_weights(pre_trained_weight_path,True)
            print('use pre-trained weights: ',pre_trained_weight_path)            
        model.compile(optimizer=Adam(lr=self.params.cnn_adam_learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

        print("training")
        filePath = os.path.join(check_folder, 'weights.{epoch:02d}.hdf5')
        checkpoint = ModelCheckpoint(filepath=filePath, monitor='loss', verbose=0, save_best_only=False,
                                     save_weights_only=False, mode='auto', period=5)
        csv_logger = CSVLogger(os.path.join(check_folder,'train.csv'))
        callbacks_list = [csv_logger,checkpoint]

        model.fit_generator(train_generator, steps_per_epoch=n_batch_per_epoch,
                            validation_data=val_generator, validation_steps=n_batch_per_epoch_val,
                            epochs=nb_epoch, callbacks=callbacks_list)

        #model.fit(data_train,label_train, batch_size=batch_size,validation_data=(data_val, label_val),
        #                     epochs=nb_epoch, callbacks=callbacks_list)
 
    def test_cnn(self,img_folder,visit_folder,out_folder='',pre_trained_weight_path=''):
        if os.path.exists(out_folder)==0:
            os.mkdir(out_folder)
        img_test=load_all_data_test(img_folder)

#        cnnModel = load_model(self.params.files['cnn_model'])
        model = get_cnn_model(self.params)
        if len(pre_trained_weight_path)>1:
            model.load_weights(pre_trained_weight_path,True)
            print('use pre-trained weights: ',pre_trained_weight_path)
        else:
            print('need pre-trained weights!!!')
            return            
        model.compile(optimizer=Adam(lr=self.params.cnn_adam_learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

        #取某一层的输出为输出新建为model，采用函数模型
        dense1_layer_model = Model(inputs=model.input,
                                            outputs=model.get_layer('fc1').output)


        AreaID=[]
        CategoryID=[]
        weigth_name=os.path.split(pre_trained_weight_path)[-1]
        Out_txt=os.path.join(out_folder,weigth_name+'_p.txt')
        f=open(Out_txt, 'w')
        print('Number of files = ', len(img_test))

        for i in range(len(img_test)):
            print('Processing files:', i/len(img_test))
            img_path=img_test[i]
            
            imageName = os.path.split(img_path)[-1]
            AreaID.append(imageName)
            visit_path=os.path.join(visit_folder,imageName[:-3]+'txt')
            #convertLab=False
            imgs,visits=GetPredictData(img_path,visit_path)#(ortho_path, dsm_path='',path_size=(256,256),overlap_ratio=0.5,convertLab=False,normalize_dsm=0,img_ioa=[]):
            ####以这个model的预测值作为输出
            dense1_output = dense1_layer_model.predict([imgs,visits])
            ######
            img_feature_path=os.path.join(out_folder,imageName[:-4]+'.npy')

            np.save(img_feature_path, dense1_output) 
            # pred = model.predict([imgs,visits])
            # pred = np.argmax(pred, axis=-1).astype('uint8')
            # pred_class=GetFinalPredbyVotes(pred)+1
            # f.write(imageName[:-4]+'\t'+'00'+str(3)+'\n')
            ##f.write("%s \t %03d\n"%(str(imageName[:-4]).zfill(6), pred_class))
        f.close()
    
def train_net(img_folder,visit_folder):
    detector=URFCBaseline(params,0)

    pre_trained_weight_path=os.path.join('')
    #pre_trained_weight_path=os.path.join('./track2-rgbh_c/','weights.03.hdf5')
    #pre_trained_weight_path=os.path.join('./weightchecks/0508-l3','weights.30.hdf5')
    #pre_trained_weight_path=os.path.join('./checkpoint_track1_unet_all_dynamic_patch512_t/','weights.10.hdf5')
    #pre_trained_weight_path=os.path.join('./checkpoint_track3-unet_all_fixed_classweight_2','weights.80.hdf5')
    #check_folder='./track2-rgb_c-morenew-new'
    check_folder='./weightchecks/rgb-256'
    #num_class=3
    #num_class=params.NUM_CATEGORIES
    detector.train_cnn(img_folder,visit_folder,check_folder,pre_trained_weight_path)
def test_net(data_folder,is_merge=False):
    detector=URFCBaseline(params,0)
    #net='unet'
    #net_name='unet_rgbh_c'
   # net_name='unet_rgb_c'
    #pre_trained_weight_path=os.path.join('./track2-rgbh_c/','weights.03.hdf5')
    pre_trained_weight_path=os.path.join('./weightchecks/rgb(256)-visit(64)-0517-l4','weights.10.hdf5')
    #pre_trained_weight_path=os.path.join('./checkpoint_track1_unet_all_dynamic_patch512_t/','weights.10.hdf5')
    #pre_trained_weight_path=os.path.join('./checkpoint_track3-unet_all_fixed_classweight_2','weights.80.hdf5')
    #check_folder='./track2-rgb_c-morenew-new'
    out_folder='./train_img_features'
    img_folder=os.path.join(data_folder,'train_img')
    visit_folder=os.path.join(data_folder,'train_visit_feature_he')
    detector.test_cnn(img_folder,visit_folder,out_folder,pre_trained_weight_path)
if __name__ == '__main__':
    # img_folder=r'G:\DataSet\UrbanClassification\data\train_img'
    img_folder=r'G:\DataSet\UrbanClassification\data-multi\train'
    visit_folder=r'G:\DataSet\UrbanClassification\data-multi\npy\train_visit'
    train_net(img_folder,visit_folder)
    #data_folder=r'G:\DataSet\UrbanClassification\data-'
    #test_net(data_folder)
  
