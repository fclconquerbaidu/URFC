import os
from multiprocessing import cpu_count

num_workers = cpu_count()-1
data_augmentation=False
use_metadata = False
visit_feature_folder=r'G:\DataSet\UrbanClassification\data\train_visit_feature_he'
num_labels =9
metadata_length = 168

cnn_last_layer_length = 256

target_img_size = (100,100)
num_channels = 3

image_format = 'jpg'

#LEARNING PARAMS
cnn_adam_learning_rate = 1e-4
cnn_adam_loss = 'categorical_crossentropy'

#DIRECTORIES AND FILES
directories = {}
directories['dataset'] = 'G:/fMoW-rgb/'
#directories['input'] = os.path.join('..', 'data', 'input')
#directories['output'] = os.path.join('..', 'data', 'output')
#directories['working'] = os.path.join('..', 'data', 'working')

directories['input'] = 'G:/TrainingData/input'
directories['output'] = 'G:/TrainingData/output'
directories['working'] = 'G:/TrainingData/working'


directories['train_data'] = os.path.join(directories['input'], 'train_data')
directories['test_data'] = os.path.join(directories['input'], 'test_data')
directories['cnn_models'] = os.path.join(directories['working'], 'cnn_models')
directories['lstm_models'] = os.path.join(directories['working'], 'lstm_models')
directories['predictions'] = os.path.join(directories['output'], 'predictions')
directories['cnn_checkpoint_weights'] = os.path.join(directories['working'], 'cnn_checkpoint_weights')
directories['lstm_checkpoint_weights'] = os.path.join(directories['working'], 'lstm_checkpoint_weights')

directories['cnn_codes'] = os.path.join(directories['working'], 'cnn_codes')

files = {}
files['training_struct'] = os.path.join(directories['working'], 'training_struct.json')
files['test_struct'] = os.path.join(directories['working'], 'test_struct.json')
files['dataset_stats'] = os.path.join(directories['working'], 'dataset_stats.json')
files['class_weight'] = os.path.join(directories['working'], 'class_weights.json')


    
category_names = ['false_detection', 'airport', 'airport_hangar', 'airport_terminal', 'amusement_park', 'aquaculture', 'archaeological_site', 'barn', 'border_checkpoint', 'burial_site', 'car_dealership', 'construction_site', 'crop_field', 'dam', 'debris_or_rubble', 'educational_institution', 'electric_substation', 'factory_or_powerplant', 'fire_station', 'flooded_road', 'fountain', 'gas_station', 'golf_course', 'ground_transportation_station', 'helipad', 'hospital', 'interchange', 'lake_or_pond', 'lighthouse', 'military_facility', 'multi-unit_residential', 'nuclear_powerplant', 'office_building', 'oil_or_gas_facility', 'park', 'parking_lot_or_garage', 'place_of_worship', 'police_station', 'port', 'prison', 'race_track', 'railway_bridge', 'recreational_facility', 'impoverished_settlement', 'road_bridge', 'runway', 'shipyard', 'shopping_mall', 'single-unit_residential', 'smokestack', 'solar_farm', 'space_facility', 'stadium', 'storage_tank','surface_mine', 'swimming_pool', 'toll_booth', 'tower', 'tunnel_opening', 'waste_disposal', 'water_treatment_facility', 'wind_farm', 'zoo']



for directory in directories.values():
    if not os.path.isdir(directory):
        os.makedirs(directory)

