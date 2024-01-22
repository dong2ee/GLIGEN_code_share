# Labelme2Caption 🚀
# by seareale(Haejin Lee)
# DATE: 2023-07-14

import concurrent.futures
import json
import pickle
from pathlib import Path

import cv2
import imagesize
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


class labelme2caption:
    ROOT = "./org"
    DEST = "./raw"

    CLASS_NAME_LIST = [
        "ulcer",
        "polyp",
        "cancer",
    ]

    def __init__(self, root=None, dest=None, classes=None):
        self.failed = []
        if root:
            labelme2caption.ROOT = root
        if dest:
            labelme2caption.DEST = dest
        if classes:
            labelme2caption.CLASS_NAME_LIST = classes

    def image_preprocess(self, img, imgsz=512):
        ## crop
        crop = img.copy()
        
        ## padding
        h, w = crop.shape[:2]
        max_ = max([h, w])
        cond = h < w
        padding = np.zeros((max_,max_,3)).astype(np.uint8)
        start_ = int(abs(h - w)/2)
        if cond:
            padding[start_:start_+h,:] = crop
        else:    
            padding[:,start_:start_+w] = crop
            
        ## resize
        resize = cv2.resize(padding, (imgsz,imgsz))

        return resize
    
    def project(self, x, projection_matrix):
        """
        x (Batch*768) should be the penultimate feature of CLIP (before projection)
        projection_matrix (768*768) is the CLIP projection matrix, which should be weight.data of Linear layer
        defined in CLIP (out_dim, in_dim), thus we need to apply transpose below.
        this function will return the CLIP feature (without normalziation)
        """
        return x @ torch.transpose(projection_matrix, 0, 1)
    
    def get_clip_feature(self, model, processor, input, is_image=False):
        which_layer_text = "before"
        which_layer_image = "after_reproject"

        if is_image:
            image = Image.fromarray(input) # Image.open(input).convert("RGB")
            inputs = processor(images=[image], return_tensors="pt", padding=True)
            inputs["pixel_values"] = inputs["pixel_values"].cuda()  # we use our own preprocessing without center_crop
            inputs["input_ids"] = torch.tensor([[0, 1, 2, 3]]).cuda()  # placeholder
            outputs = model(**inputs)
            feature = outputs.image_embeds
            if which_layer_image == "after_reproject":
                feature = self.project(feature, torch.load(str(Path(__file__).parent / "projection_matrix")).cuda().T).squeeze(0)
                feature = (feature / feature.norm()) * 28.7
                feature = feature.unsqueeze(0)
        else:
            inputs = processor(text=input, return_tensors="pt", padding=True)
            inputs["input_ids"] = inputs["input_ids"].cuda()
            inputs["pixel_values"] = torch.ones(1, 3, 224, 224).cuda()  # placeholder
            inputs["attention_mask"] = inputs["attention_mask"].cuda()
            outputs = model(**inputs)
            if which_layer_text == "before":
                feature = outputs.text_model_output.pooler_output

        return feature[0]

    def convertJson(self, json_file, txt_path):
        w, h = imagesize.get(str(json_file.parent / (json_file.stem + ".jpg")))
        cond = h < w
        pad = int(abs(h - w)/2)
        w_ = w + (0 if cond else 2*pad)
        h_ = h + (2*pad if cond else 0)
        

        # load json file
        with open(json_file, "r") as js:
            cur_json = json.load(js)
            
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        with open(txt_path, "w") as f:
            # check object exists
            if len(cur_json["shapes"]) == 0:    
                return False # normal images
            
            for anno in cur_json["shapes"]:
                diagnosis = anno["label"]
                print(diagnosis)
                class_num = labelme2caption.CLASS_NAME_LIST.index(diagnosis)
                print(class_num)
                
                points = anno["points"]

                x1, y1 = map(int, np.min(points, axis=0))
                x2, y2 = map(int, np.max(points, axis=0))
                x1 += (0 if cond else pad)
                x2 += (0 if cond else pad)
                y1 += (pad if cond else 0)
                y2 += (pad if cond else 0)
                
                x1, x2 = np.clip([x1, x2], 0, w_)
                y1, y2 = np.clip([y1, y2], 0, h_)

                # calculate components
                bbox_center_x = ((x2 + x1) / 2)
                bbox_center_y = ((y2 + y1) / 2)
                
                bbox_width = x2 - x1

                if bbox_width < 0:
                    print("----------------------")
                    print(json_file.name)
                    print(points)
                    print(x1, y1, x2, y2)
                    print("----------------------")

                bbox_height = y2 - y1

                # normalize the coordinates
                bbox_center_x /= w_
                bbox_center_y /= h_
                bbox_width /= w_
                bbox_height /= h_

                bbox_width = min(1.0, bbox_width)
                bbox_height = min(1.0, bbox_height)

                f.write(f"{class_num} {bbox_center_x} {bbox_center_y} {bbox_width} {bbox_height}\n") 

        return True

    def get_caption(self, lesion_classes, bbox_list):
        lesion_caption = []
        for x, (_, _, b_w, b_h) in zip(lesion_classes, bbox_list):          
            lesion_size = "full size" if b_w * b_h >= 0.81 else "large" if b_w * b_h >= 0.25 else "medium" if b_w * b_h >= 0.04 else "small"
            lesion_caption.append(" ".join([lesion_size, x.lower()]))
        lesion_caption = ", ".join(lesion_caption)

        final_caption = f"{lesion_caption if len(lesion_classes) else 'no polyp'} on a large intestine"

        return final_caption

    def get_labels(self, txt_path):
        caption_list = []
        bbox_list = []
        with open(str(txt_path), "r") as f:
            read_lines = f.readlines()
            for r_line in read_lines:
                category_id = int(r_line.split()[0])
                class_name = labelme2caption.CLASS_NAME_LIST[category_id]
                if class_name:
                    caption_list.append(class_name.lower())
                    ###
                    bbox_list.append([float(x) for x in r_line.split()[1:]])

        return caption_list, bbox_list

    def thread_run(self, pbar, img_set, root, dest, task):
        if task == "embedding":
            ##### for embedding task
            version = "openai/clip-vit-large-patch14"
            model = CLIPModel.from_pretrained(version).cuda()
            processor = CLIPProcessor.from_pretrained(version)
            ###
        #print(len(img_set))
        for img_path in img_set:
            json_file = img_path.parent / (img_path.stem + ".json")
            img_dest_path = Path(dest) / Path(*(img_path.parent.parts[len(Path(root).parts):])) / img_path.name
            ## image preprocess
            if task == "img":
                img = cv2.imread(str(img_path))
                img = self.image_preprocess(img)
                img_dest_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(img_dest_path), img)
                
            ## get bbox info
            if task == "bbox": 
                json_file = img_path.parent / (img_path.stem + ".json")
                txt_dest_path = img_dest_path.parent / (img_dest_path.stem + ".txt")
                if json_file.exists():
                    bbox_results = self.convertJson(json_file, txt_dest_path)

                else:
                    txt_dest_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(txt_dest_path, "w") as f:
                        pass
               
            
            ## get caption
            if task == "caption": 
                txt_dest_path = img_dest_path.parent / (img_dest_path.stem + ".txt")
                class_list, bbox_list = self.get_labels(txt_dest_path)

                img = cv2.imread(str(img_dest_path))[:,:,::-1]
                caption = self.get_caption(class_list, bbox_list)
                
                caption_path = img_dest_path.parent / (img_dest_path.stem + "_caption.txt")
                caption_path.parent.mkdir(parents=True, exist_ok=True)
                with open(str(caption_path), "w") as f:
                    f.write(caption)
            
            ## get embedding
            if task == "embedding": 
                txt_dest_path = img_dest_path.parent / (img_dest_path.stem + ".txt")
                class_list, bbox_list = self.get_labels(txt_dest_path)
                
                #### Generate Embedding ####################
                h, w = imagesize.get(img_dest_path)
                img = cv2.imread(str(img_dest_path))[:,:,::-1]

                embedding_list = {
                    "text_embedding_before": [],
                    "image_embedding_after": [],
                }
                for class_name, (c_x, c_y, b_w, b_h) in zip(class_list, bbox_list):
                    x1, y1, b_w, b_h = [
                        int((c_x - (b_w / 2)) * w),
                        int((c_y - (b_h / 2)) * h),
                        int(b_w * w),
                        int(b_h * h),
                    ]
                    box_image = img[max(0, y1) : min(y1 + b_h, h), max(0, x1) : min(x1 + b_w, w)]

                    image_embedding_after = self.get_clip_feature(model, processor, box_image, is_image=True)
                    text_embedding_before = self.get_clip_feature(model, processor, class_name.lower(), is_image=False)
                
                    embedding_list["text_embedding_before"].append(text_embedding_before)
                    embedding_list["image_embedding_after"].append(image_embedding_after)

                emb_dest_path = img_dest_path.parent / (img_dest_path.stem + ".pk")
                emb_dest_path.parent.mkdir(parents=True, exist_ok=True)
                pbar.set_description(f"{emb_dest_path.name}")
                with open(str(emb_dest_path), "wb") as f:
                    pickle.dump(embedding_list, f)
                #############################################
                    
            pbar.update(1)

    def run(self, task="img"):
        img_list = sorted(list(Path(labelme2caption.ROOT).rglob("*.jpg")))
        
        print(f"{labelme2caption.ROOT}: {len(img_list)}")

        img_split_list = np.array_split(img_list, 4)
        
        with tqdm(img_list) as pbar:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                
                results = {
                    executor.submit(self.thread_run, pbar, img_set, 
                                    labelme2caption.ROOT, 
                                    labelme2caption.DEST, task): img_set for img_set in img_split_list
                }