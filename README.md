# yolo_from_scratch
Sıfırdan Adım Adım PyTorch Kullanılarak YOLO İmplementasyonu Google Colab

*Yolo algoritması arkasındaki fikir
*Model Mimarisi
*Kayıp Fonksiyonu
Bu üç başlığı anlamamız implementasyon yapabilmemiz için oldukça önemlidir.

YOLO PASCAL VOC Data setiyle eğitilmiştir. Bu daha set içerisinde 20 farklı sınıf bulundurmaktadır.

![Alt text](https://miro.medium.com/max/504/1*VSa5Fjrz0oNM7_iYFzyAbg.png "img0")

Ancak son zamanlarda makalelerde VOC yerine COCO data seti daha çok kullanılmaktadır.

YOLO Algoritması


YOLO algoritmasında amacımız resimdeki nesneleri birer doğru bounding box ile tespit etmek ve bu nesnelerin ne olduğunu bulmaktır.



![Alt text](https://miro.medium.com/max/565/1*Me6Z8ETRcV0eiiidM1iXEQ.png "img1")

Bunun için resmimizi SxS hücreye böleriz. Makalede S=7 alınmış ancak biz anlatım kolaylığı olması açısından S=3 alacağız.

![Alt text](https://miro.medium.com/max/568/1*PpuDxa6QJe7nj6RGRz-8iA.png "img1")

Her hücre karşılık gelen bounding boxla bir tahmin çıkarıyor.
Yukarıdaki resme baktığımızda her iki köpeğin de birdden fazla hücrede yer aldığını görüyoruz.
Biz her nesne için bir bounding box olmasını istiyoruz.
Bu nesneyi bulmakla sorumlu bir hücre seçeceğiz. Sorumlu olan hücre bu nesnenin orta noktasını bulunduran hücre olacaktır.
