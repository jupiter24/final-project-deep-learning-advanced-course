from glob import glob
from pathlib import Path
import os

class_dir = '/home/deep-learning-advanced-course/data/train/'
class_dir_val = '/home/deep-learning-advanced-course/data/val/'
class_names = ['n01440764', 'n01990800', 'n02108915', 'n02490219', 'n02951585', 'n03444034', 'n03692522', 'n04254680', 'n04487081', 'n04592741',
'n01692333', 'n02025239', 'n02112137', 'n02643566', 'n02999410', 'n03461385', 'n03792782', 'n04258138', 'n04536866', 'n07715103',
'n01773797', 'n02058221', 'n02132136', 'n02704792', 'n03063689', 'n03594945', 'n03792972', 'n04273569', 'n04540053', 'n07730033',
'n01855032', 'n02094433', 'n02229544', 'n02865351', 'n03100240', 'n03661043', 'n04201297', 'n04317175', 'n04548362', 'n07749582',
'n01984695', 'n02105412', 'n02395406', 'n02869837', 'n03388183', 'n03680355', 'n04252077', 'n04392985', 'n04584207', 'n07875152']
for class_name in class_names:
    image_files = glob(class_dir+class_name+'/*')
    path = Path((class_dir + class_name).replace('train', 'train_50classes'))
    path.mkdir(parents=True, exist_ok=True)
    for image_file in image_files:
        os.rename(image_file, image_file.replace('train', 'train_50classes'))
    image_files = glob(class_dir_val + class_name + '/*')
    path = Path((class_dir_val + class_name).replace('val', 'val_50classes'))
    path.mkdir(parents=True, exist_ok=True)
    for image_file in image_files:
        os.rename(image_file, image_file.replace('val', 'val_50classes'))
