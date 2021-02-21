# -*- coding: utf-8 -*-
"""YOLO_dataset.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ogzqLpSWzGZ1dC5LtGThPMjN-ZlHlzgn
"""

"""
Creates a Pytorch dataset to load the Pascal VOC dataset
"""

import torch
import os
import pandas as pd
from PIL import Image


class VOCDataset(torch.utils.data.Dataset): #bilgileri buradan alacağız. (inherit)
    def __init__(
        self, csv_file, img_dir, label_dir, S=7, B=2, C=20, transform=None, #resminyeri,labelınyeri,split size,number of bounding boxes,number of classes)
      
    ):
        self.annotations = pd.read_csv(csv_file) #csv dosyasını okuyoruz
        self.img_dir = img_dir 
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        return len(self.annotations) #annotation= açıklama, dipnot

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1]) #labelın yolunu buluyoruz. 1 dedik çünkü text dosyasının olduğu sütun o
        boxes = [] #boş bir liste
        with open(label_path) as f: #text dosyasını açacağız
            for label in f.readlines():
                class_label, x, y, width, height = [ #class integer olacak ve biz de bunun int olmasını istiyoruz, float inte eşit değilse stringe çevirip integerı alıyoruz
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n", "").split()
                ]

                boxes.append([class_label, x, y, width, height])
             #belirli satırı alıyoruz floatları integera çeviriyoruz,sonra bu halini listeye ekliyoruz (append )
             #label kısmını tamamladık
        #resim kısmına geçiyoruz
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])#resmin yolunu alıyoruz sonra da resim dosyasının ismini alıyoruz buraya 0 yazarak
        image = Image.open(img_path) #pil burada kullanılıyor
        boxes = torch.tensor(boxes) #tensore çeviriyoruz

        if self.transform:
            # image = self.transform(image)
            image, boxes = self.transform(image, boxes)
        #tensore çevirdik onu tekrar labela çeviriyoruz
        # Convert To Cells
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B)) #(S,S,30)
        for box in boxes: #yukarıdaki label matrixe uyması için her şeyi dönüştürüyoruz.Tüm resme bağımlı ama biz her bir bounding box hangi hücrede onu bulup tüm resme değil hücreye bağımlı hale getiriyoruz 
            class_label, x, y, width, height = box.tolist() 
            class_label = int(class_label) #classların integer olduğundan emin oluyorum
            #hangi hücre satırı ve sütununa ait onu belirliyoruz
            # i,j represents the cell row and cell column
            i, j = int(self.S * y), int(self.S * x) #satır,sütun -- ölçeklemek için hücre sayısı ile çarpacağız--0 ile 1 arasında bir oran olacak
            x_cell, y_cell = self.S * x - j, self.S * y - i   #hücreye bağımlı x ve y hücrede bulunduğu parçayı çıkararak buluyoruz

            """
            Calculating the width and height of cell of bounding box,
            relative to the cell is done by the following, with
            width as the example:
            
            width_pixels = (width*self.image_width)
            cell_pixels = (self.image_width)
            
            Then to find the width relative to the cell is simply:
            width_pixels/cell_pixels, simplification leads to the
            formulas below.
            """
            width_cell, height_cell = (
                width * self.S, #genişlik ve yükseklik tüm resme bağımlı  hücreye bağlı olmasını istersek sadece hücre sayısıyla yeniden boyutlandırıyoruz
                height * self.S,
            )#buraya kadar bu fonksiyonda hepsi hücreye bağlı koordinat değerlerimizi buluyor

            # If no object already found for specific cell i,j
            # Note: This means we restrict to ONE object
            # per cell!
            if label_matrix[i, j, 20] == 0: #hiçbir obje yoksa bunu kontrol ediyoruz
                # Set that there exists an object
                label_matrix[i, j, 20] = 1 #gelecekte 1 olacak diye bire eşitliyoruz

                # Box coordinates
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )

                label_matrix[i, j, 21:25] = box_coordinates

                # Set one hot encoding for class_label
                label_matrix[i, j, class_label] = 1 #hangi sınıf olduğunu belirliyoruz

        return image, label_matrix


        #utilsi yapmayacağım daha önceki videolarda anlattım dedi sırada train var dedi