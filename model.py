# -*- coding: utf-8 -*-
"""YOLO_model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1h79tuGf4NEAv7rcmbqJ_JR60PbGzUcFS
"""

""" 
Implementation of Yolo (v1) architecture
with slight modification with added BatchNorm.
"""

import torch
import torch.nn as nn

""" 
Information about architecture config:
Tuple is structured by (kernel_size, filters, stride, padding) 
"M" is simply maxpooling with stride 2x2 and kernel 2x2
List is structured by tuples and lastly int with number of repeats
"""
#(kernel_size, filters, stride, padding)
#"M" maxpoolu ifade ediyor 2x2lik kernellı ve 2x2lik stridelı (adım)

architecture_config = [
    #modelin mimarisinden yola çıkarak bu kısım yazıldı
#Tuple = (kernel_size, filters, stride, padding)
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),

    #liste: 
    "M",
    #Liste: tuplelar, işlemin kaç kez tekrar edildiğini söyleyen sayı
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],#4 burada işlemin kaç kez yapıldığını söylüyor her iki tuple için 
    (1, 512, 1, 0), 
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],#2 burada işlemin kaç kez yapıldığını söylüyor
    (3, 1024, 1, 1),#4 tane conv layer
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]


class CNNBlock(nn.Module): #CNNİ burada yapıyoruz çok kez kullanacağız
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs) #bias kullanmıyoruz batchnorm kullanıyoruz
        self.batchnorm = nn.BatchNorm2d(out_channels) #makalede batchnorm yok çünkü o zamanlar batchnorm bulunmamıştı ama biz burada kullanıyoruz.
        self.leakyrelu = nn.LeakyReLU(0.1) #eğim için 0.1 kullandık

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))


class Yolov1(nn.Module): #yolov1 kullanıyoruz 
    def __init__(self, in_channels=3, **kwargs): #rgb olduğu için kanal sayısı 3 
        super(Yolov1, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture) #conv katman darknet olarak adlandırılarak kullanılıyor
        self.fcs = self._create_fcs(**kwargs) #fully connected katman, kwargs  keyword argument 

    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1)) #örnekleri düzleştirmek istmediğimiz için 1 aldık. Düzleştirip (flatten) fully connecteda gönderiyoruz

    def _create_conv_layers(self, architecture):
        layers = [] #boş liste
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == tuple: #tipi tuplesa cnnblocka bir ekle 
                layers += [
                    CNNBlock(
                        in_channels, x[1], kernel_size=x[0], stride=x[2], padding=x[3],
                    )                #x[1] out channels
                ]
                in_channels = x[1]

            elif type(x) == str: #tipi stringse maxpool ekle
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

            elif type(x) == list: #listeyse her liste 2 tane conv katmana sahip ve sırayla tekrar ediyor
                conv1 = x[0] #Tuple
                conv2 = x[1] #Tuple
                num_repeats = x[2] #Integer

                for _ in range(num_repeats):
                    layers += [ #birinci
                        CNNBlock(
                            in_channels,
                            conv1[1],
                            kernel_size=conv1[0],
                            stride=conv1[2],
                            padding=conv1[3],
                        )
                    ]
                    layers += [ #ikinci
                        CNNBlock(
                            conv1[1],
                            conv2[1],
                            kernel_size=conv2[0],
                            stride=conv2[2],
                            padding=conv2[3],
                        )
                    ]
                    in_channels = conv2[1] #sonraki conv layer girişi bizim çıkışımıza eşit olacak

        return nn.Sequential(*layers)#listeyi açıyor ve nn sequentiale çeviriyor

    def _create_fcs(self, split_size, num_boxes, num_classes): #fully connected layer oluşturma
        S, B, C = split_size, num_boxes, num_classes

        # In original paper this should be
        # nn.Linear(1024*S*S, 4096),
        # nn.LeakyReLU(0.1),
        # nn.Linear(4096, S*S*(B*5+C))

        return nn.Sequential( 
            nn.Flatten(), #her şeyi düzleştirdik çünkü lineer yapcaz
            nn.Linear(1024 * S * S, 496), #kısa sürsün diye 496 yaptık makalede 4096
            nn.Dropout(0.0),#default olarak verdik
            nn.LeakyReLU(0.1), #eğim 0,1 
            nn.Linear(496, S * S * (C + B * 5)), #tekrar şekillendirme linnerden sonra (S,S,30) C+b*5=30
        )



####test etmek için

def test (S=7,B=2,C=20):
  model = Yolov1(split_size=S,num_boxes=B,num_classes=C)
  x=torch.randn((2,3,448,448))
  print(model(x).shape)
test()