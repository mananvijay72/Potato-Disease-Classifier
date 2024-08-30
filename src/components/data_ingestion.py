
import tensorflow as tf
import os
import numpy
from PIL import Image




def convert_jpg_to_jpeg(file_path):

    with Image.open(file_path) as img:
        # Define the new file path with .jpeg extension
        new_file_path = os.path.splitext(file_path)[0] + '.jpeg'
        
        # Save the image in JPEG format
        img.convert('RGB').save(new_file_path, 'JPEG')
        
        # Optionally, remove the original .jpg file
        os.remove(file_path)
        
        print(f"Converted {file_path} to {new_file_path}")


class LoadData:

    def __init__(self):
        pass

    def clean_data(self,data_dir: str):
        
        image_exts = ['jpeg','png', 'bmp', 'jpg']

        for image_class in os.listdir(data_dir): 
            for image in os.listdir(os.path.join(data_dir, image_class)):
                image_path = os.path.join(data_dir, image_class, image)
                try: 
                    tip = image_path.split(".")[-1].lower()
                    if tip == 'jpg':
                        convert_jpg_to_jpeg(image_path)
                        
                    if tip not in image_exts: 
                        print('Image not in ext list {}'.format(image_path))
                        os.remove(image_path)
                        print("removed")
                except Exception as e: 
                    print('Issue with image {}'.format(image_path))
                    os.remove(image_path)
    

    def ingest_data(self, data_dir:str):

        #making sure files have proper format
        self.clean_data(data_dir)

        #batch_size=32,image_size=(256, 256)
        data = tf.keras.utils.image_dataset_from_directory(data_dir)
        
        return data

