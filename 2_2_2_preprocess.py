import cv2
import os
import natsort
from pathlib import Path
import json
import numpy as np
from PIL import Image
import shutil
from labelme2caption import labelme2caption

path = Path('/raid/bumsu/nia_data/2_2_2/origin_data')
new_caption = True

def create_crop_json():
    image_list = sorted(list(Path(path).rglob("*.jpg")))
    label = []
    if not (path / 'rmletter').exists():
        os.mkdir(path / 'rmletter')
    
    for img_path in image_list:
        json_path = Path(img_path.parent / (str(img_path.stem)+".json"))
        json_crop_path = path / 'rmletter' / (str(img_path.stem)+'.json')
        img_crop_path = path / 'rmletter' / (str(img_path.stem)+'.jpg')
        
        if img_path in list((path / 'rmletter').rglob("*.jpg")) and json_path in list(Path(path / 'rmletter').rglob("*.json")):
            continue
        
        image = Image.open(img_path)
        img_np = np.array(image)

        remove_x = int(img_np.shape[1]*0.05)
        img_np = img_np[:,remove_x:,:]
   
        row_size = len(img_np[0])
        col_size = len(img_np)
        remove_loc = int(row_size / 512 * 80)

        for r in range(row_size):
            if r < remove_loc:
                img_np[:(remove_loc-r), r, :] = 0
                img_np[(col_size-remove_loc+r):, r, :] = 0

            elif r > row_size-remove_loc:
                img_np[:(remove_loc-row_size+r), r, :] = 0
                img_np[(col_size-remove_loc+row_size-r):, r, :] = 0
                
        img_pil = Image.fromarray(img_np)
        img_pil.save(img_crop_path)        

        if json_path.exists():
            with open(json_path, "r") as json_file:
                json_load = json.load(json_file)
                
                for i,anno in enumerate(json_load["shapes"]):
                    points = anno["points"]
                    x1, y1 = np.min(points, axis=0)
                    x2, y2 = np.max(points, axis=0)
                    
                    x1_crop = x1 - remove_x
                    x2_crop = x2 - remove_x

                    json_load["shapes"][i]["points"] = [[x1_crop, y1], [x2_crop, y1], [x2_crop, y2], [x1_crop, y2]]
                    
                with open(json_crop_path, 'w') as json_crop_file:
                    json.dump(json_load, json_crop_file, indent=2)
    print('all new json files and images are saved')

def get_new_caption():
    caption_list = sorted(list(path.rglob("*.txt")))
    key_list = []
    caption_dict = {}
    case_dicts = []

    label = []
    for n, caption_file in enumerate(caption_list):
        
        case_dict = {}
        case_dict.setdefault('file_name')
        case_dict['file_name'] = str(caption_file)

        with open(caption_file, "r") as f:
            lines = f.readlines()
            
            for line in lines:
                space_split = line.split(' ')
                for i, splitted in enumerate(space_split):
                    if '/' not in splitted:
                        space_split[i-1] = space_split[i-1] + ' ' + space_split[i]
                        space_split = space_split[:-1]
                    elif 'exisit' in splitted:
                        space_split[i-1] = space_split[i-1] + ' ' + 'exisit'
                        space_split[i] = space_split[i][6:]

                for dic in space_split:
                    key = dic.split('/')[0].strip()
                    value = dic.split('/')[1].strip()
                    
                    case_dict.setdefault(key)
                    case_dict[key] = value

                    if key not in caption_dict.keys():
                        caption_dict.setdefault(key)
                        caption_dict[key] = []
                        caption_dict[key].append(value)
                        
                    elif value not in caption_dict[key]:
                        caption_dict[key].append(value)
        
        case_dicts.append(case_dict)
        
    diagnosis_key = ['polyp_shape', 'diagnosis', 'polyp_size', 'polyp_path']
    object_key = ['cap', 'situation', 'device']
    location_key = ['location', 'age']
    view_key = ['view', 'lesion_view']

    if not (path.parent / 'preprocessed').exists():
        os.mkdir(path.parent / 'preprocessed')
        
    for i, caption_file in enumerate(caption_list):
        json_path = path / (str(caption_file.stem[:-8])+".json")
        json_crop_path = path / 'rmletter' / (str(caption_file.stem[:-8])+'.json')

        is_diagnosis = False
        
        for j, case in enumerate(case_dicts):
            if case['file_name'] == str(caption_file):
                case_num = j
        new_caption_file = path.parent / 'preprocessed' / (str(caption_file.stem) + '.txt')
        case_dict = case_dicts[case_num]
        sentence = 'An image of '
        
        if view_key[0] in case_dict.keys():
            sentence = sentence + case_dict[view_key[0]]
            
            if view_key[1] in case_dict.keys():
                sentence = sentence + ' and ' + case_dict[view_key[1]] + ' view of lesion.'
            else:
                sentence += '.'
        
        elif view_key[1] in case_dict.keys():
            sentence = sentence + ' An image of ' + case_dict[view_key[1]] + ' view of lesion.'

        if diagnosis_key[1] in case_dict.keys():
            label.append(diagnosis_key[1])
            sentence += ' There is '
            if caption_file.stem[:-8] + '.json' not in os.listdir(path):
                sentence += 'no '
            else: is_diagnosis = True
            sentence += case_dict[diagnosis_key[1]]
        
        if diagnosis_key[2] in case_dict.keys():
            sentence += ' '
            if case_dict[diagnosis_key[2]] == '<5mm':
                sentence += 'not more than 5mm'
            elif case_dict[diagnosis_key[2]] == '5~9mm':
                sentence += 'not less than 5mm and not more than 9mm'
            elif case_dict[diagnosis_key[2]] == '>=10mm':
                sentence += 'greater than 10mm'
        
        if object_key[1] in case_dict.keys():
            sentence = sentence + ' with ' + case_dict[object_key[1]]
            
        sentence += '.'
        
        with open(new_caption_file, "a") as f:
            f.write(sentence)
            
        if json_path.exists():
            with open(json_path, "r") as json_file:
                json_load = json.load(json_file)
                
                for i,anno in enumerate(json_load["shapes"]):
                    json_load["shapes"][i]["label"] = case_dict[diagnosis_key[1]]
                    
                with open(json_crop_path, 'w') as json_crop_file:
                    json.dump(json_load, json_crop_file, indent=2)
                    
    print('all new captions are saved')
    
create_crop_json()

if new_caption:
    get_new_caption()
else:
    pass

DATA_PATH = str(path / 'rmletter')
SAVE_PATH = str(path.parent / 'preprocessed')
converter = labelme2caption(root=DATA_PATH, dest=SAVE_PATH)
converter.run(task="img")
converter.run(task="bbox")
converter.run(task="embedding")

image_list = sorted(list(Path('/raid/bumsu/nia_data/2_2_2/preprocessed').rglob("*.jpg")))
caption_list = sorted(list(Path('/raid/bumsu/nia_data/2_2_2/preprocessed').rglob("*_caption.txt")))
txt_list = sorted(list(Path('/raid/bumsu/nia_data/2_2_2/preprocessed').rglob("*).txt")))
embedding_list = sorted(list(Path('/raid/bumsu/nia_data/2_2_2/preprocessed').rglob("*.pk")))

print(len(image_list))
print(len(caption_list))
print(len(txt_list))
print(len(embedding_list))