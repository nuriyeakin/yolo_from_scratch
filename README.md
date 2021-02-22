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

![Alt text](https://miro.medium.com/max/568/1*PpuDxa6QJe7nj6RGRz-8iA.png "img2")

Her hücre karşılık gelen bounding boxla bir tahmin çıkarıyor.
Yukarıdaki resme baktığımızda her iki köpeğin de birdden fazla hücrede yer aldığını görüyoruz.
Biz her nesne için bir bounding box olmasını istiyoruz.
Bu nesneyi bulmakla sorumlu bir hücre seçeceğiz. Sorumlu olan hücre bu nesnenin orta noktasını bulunduran hücre olacaktır.

![Alt text](https://miro.medium.com/max/564/1*kBan-ZymB1EMM2jp--gigw.png "img3")

Resimdeki mavi noktalar bu nesnelerin orta noktalarını göstermektedir.

![Alt text](https://miro.medium.com/max/605/1*221qkggyjvtqnEpuIf8GZA.png "img4")

Oklarla orta noktaların bulunduğu hücreler gösterilmektedir.
İlk orta noktamızın bulunduğu hücreye yakından baktığımızda:

![Alt text](https://miro.medium.com/max/405/1*hfehQ06AqnUOxJdryh-j9g.png "img5")

Bu şekilde görmekteyiz.

Ayrıca unutmamamız gereken bir konu var.

Hücrenin köşesini orijin olarak kabul ederiz ve (0,0) değerini alır. 

Yatay olarak hareket ettiğimizde x değerimiz dikey olarak hareket ettiğimizde y değerimiz artacaktır.

Her bir çıktı ve etiket hücreye bağımlı(relative) olacaktır.

![Alt text](https://miro.medium.com/max/217/1*xmGvezxJnsCxzIT1m33egw.png "img6")

Ayrıca her bir bounding box yukarıdaki gibi koordinatlarını belirten noktalara sahip olacaktır.

![Alt text](https://miro.medium.com/max/184/1*Ccm46G3WTAqLIw_xhesv2g.png "img7")

Bu şekilde gösterildiğinde ise x ve y değerleri nesnemizin orta noktasını vermektedir.
W genişlik h yükseklik değerleridir.

![Alt text](https://miro.medium.com/max/188/1*lTuLMMd6hQr3Mag6aHXozw.png "img8")

Daha ayrıntılı incelememiz gerekirse:

X değerimiz 0 ile 1 arasında olacaktır ve orta noktamızın hücrede orijine göre ne kadar sağda yer aldığını gösterir.

Y değerimiz de 0 ile 1 arasında olacaktır ve orta noktamızın orijine göre ne kadar aşağıda yer aldığını gösterir.

W değerimiz genişlik değeridir ve eğer nesnemiz hücreden daha genişse 1’den büyük değer alabilir.

H değerimiz yükseklik değeridir ve nesnemiz hücreden daha uzunsa 1’den büyük değer alabilir.

![Alt text](https://miro.medium.com/max/355/1*y-2NKFhbtmHrkmWPKwPybg.png "img9")

Bu nesnemize baktığımızda x,y,w ve h değerlerimiz tahmini olarak bu şekilde çıkmaktadır.

![Alt text](https://miro.medium.com/max/281/1*QwI0uCsRvzo9ikfCpvkU_g.png "img10")

Hücreye bağlı etiket değerimize baktığımızda:

![Alt text](https://miro.medium.com/max/605/1*xtmRSA9OzLfO97vM-Ok5nA.png "img11")

C değerleri 20 farklı sınıfımızı temsil etmektedir.

P değeri hücrede nesne olup olmadığını belirten olasılık değeridir ve nesne varsa 1 yoksa 0 şeklinde düşünülür.

x,y,w,h değerlerimiz bounding boxımızın koordinatlarını vermektedir.

Hücreye bağlı tahmin değerimiz de benzer olacaktır ancak 2 bounding box koordinatlarını içerecektir.

![Alt text](https://miro.medium.com/max/605/1*vsg7FmiXJD70vSKj3u_hfg.png "img12")

Bunun amacı nesnemize göre bounding box şeklimizi belirlemektir. Yani nesne uzunsa ona uygun, nesne genişse ona uygun bounding box tahminlerini yapabilmektir.


![Alt text](https://miro.medium.com/max/600/1*Vip6m0_RSfvlwTP5raoiXA.png "img13")

Şunu unutmamalıyız ki bir hücre yalnızca 1 nesne için bounding box tespit edebilir. Bu YOLO’nun sınırlamalarından biridir.

Eğer daha fazla bounding box istiyorsak daha ince gridlere böleebiliriz.

Yani bizim örnek olarak gösterdiğimiz 3x3 değil de makaledeki gibi 7x7 kullanabiliriz.

Buraya kadar bahsettiğimiz her bir değer hücreye özel, bağımlı(relative) değerlerdir.

Bir resim için hedef şekli:

![Alt text](https://miro.medium.com/max/139/1*kQBUZDXiAwoCBwXtvIvtGw.png "img14")

25 değerinin 20’si sınıflarmızı, 1’i olasılık skorumuzu, 4’ü de bounding boxımızın koordinat değerlerini (x,y,w,h) vermektedir.

Bir resim için tahmin şekli:

![Alt text](https://miro.medium.com/max/139/1*kQBUZDXiAwoCBwXtvIvtGw.png "img15")

30 değerinin 25’i hedefle aynı değerlerdir ancak 1 tane fazla olasılık skoru ve bounding boxımızın koordinat değeri olan 4 değeri de(x,y,w,h) içerdiği için toplam 30 olmaktadır.


YOLO Mimarisi

![Alt text](https://miro.medium.com/max/605/1*fc559AJxPZ7UWb5_xSGGyg.png "img16")

Mimari modeliz için en önemli kısımdır.

Input değerimize baktığımızda 448x448x3 olduğunu görüyoruz. 3 bize RGB bir resmimiz olduğunu söylüyor.

448ler de genişlik ve yükseklik değerlerini veriyor.

![Alt text](https://miro.medium.com/max/161/1*b5M296HnxNNg8AclEkZsDg.png "img17")

7x7 kernel değeri

64 output filtresi

2 stride(adım) değeri

Sonrasında 2x2 maxpool stride değeri 2 olan

Buradaki 2 olan stride değerimiz resim boyutumuzu yarıya indirecektir.

Ayrıca maxpooldaki 2 olan stride değerimiz de değerimiz resim boyutumuzu yarıya indirecektir.

Bu nedenle 448’den 112’ye düştüğünü görüyoruz.

![Alt text](https://miro.medium.com/max/141/1*MwrjMgTlKsvE40PSBJpRLQ.png "img18")

3x3 kernel değeri

192 output filtresi

Sonrasında 2x2 maxpool stride değeri 2 olan

Stride ile 112’den 56’ya düşüyor değerimiz.

İşlemler benzer şekilde devam ediyor hepsini ayrıntılı olarak anlatmayacağım.

![Alt text](https://miro.medium.com/max/103/1*1oXj_MXwS6avTPP6h35CUg.png "img19")

Ancak buradaki x4 bu işlemin 4 tekrar edildiğini söylüyor.

Burada 2 tane ek conv layer yapılıyor ama input değerimizi değiştirmiyor.

![Alt text](https://miro.medium.com/max/161/1*cmZlZwSLO-EQL7c-KNsbXw.png "img20")

Burada da fully connected layerla lineer hale getirip sonrasında tekrar fully connectrd layer uygulayarak yeniden şekillendiriliyor.

Sonuçta 7x7x30 şeklinde bir çıktı elde ediyoruz. Buradaki 7 değerleri yukarıda bahsettiğimiz s değeri yani resmi böleceğimiz grid sayısıdır. 30 da yukarıda bahsettiğimiz tahmin şeklimizdeki değerlerdir.


![Alt text](https://miro.medium.com/max/218/1*kt-LmjKaQAaAzW_r7cBxHQ.png "img21")

İmplementasyonumuzda oluşturup kullanacağımız kod dosyaları:

model.py

loss.py

train.py

utils.py

dataset.py

model.py ile başlayalım.

![Alt text](https://miro.medium.com/max/605/1*ddTLoLprul34nna_JNtWTg.png "img22")

Koddaki bu kısım mimariden yola çıkılarak yazılmıştır.

Kayıp (Loss) Fonksiyonu

![Alt text](https://miro.medium.com/max/553/1*DKHuV1nEtn4viq9lkaaq8A.png "img23")

![Alt text](https://miro.medium.com/max/378/1*uKuIYbW7UMvK27-myT5meA.png "img24")

Denklemimizin bir numaralı kısmına bakarsak:

Önce karelerini alıp sonra toplayacağız.

X ve y değerleri kutuların (orta noktaların) koordinat değerleri

λ sabitini kullanıyoruz çünkü yüksek değere öcelik vermek istiyoruz. Bounding boxları doğru bulduğumuzdan emin olabilmek için de bu değeri 5 ile çarpacağız. Yani λ=5.

Birinci toplam değerimiz nesne için kullanılıyor.

![Alt text](https://miro.medium.com/max/31/1*_fQQm_JObVDM9H1j-lirjQ.png "img25")

İkinci toplam değerimiz hedefimizdeki bounding boxın IoU’su maksimum olacak şekilde seçiyor.
Yani burada sadece en iyi tahmini eğitiyoruz. Yoksa sıfır olacak.

![Alt text](https://miro.medium.com/max/33/1*rsm68glVZ45uQMJIyzRYuQ.png "img26")

Bu değer identity fonksiyonu. Hücrede box olup olmamasına göre 1 ya da 0 olacak yani nesne varsa 1 yoksa 0

Denklemimizin iki numaralı kısmına bakarsak:

![Alt text](https://miro.medium.com/max/496/1*ag5Z4mahlxhMku2i-rMAaQ.png "img27")

W ve h değerleri genişlik ve yükseklik değerlerimiz.

Burada karekök alarak küçük bounding boxlara da büyük bounding boxlar kadar öncelik veriyoruz.

Denklemin 1 ve 2 numaralı kısımları bounding boxların koordinatları için

Denklemimizin üç numaralı kısmına bakarsak:

![Alt text](https://miro.medium.com/max/240/1*O-FFCQe6GQ_43u-zA0ZFNw.png "img28")

C olasılık değerimiz hücrede bounding box var mı yok mu ona bakıyor.

Yine en yüksek IoU’lu olanı alacağız bu aldığımız değer tahminden sorumlu olacak.

Eğer bir nesne varsa mavi ile altı çizili olan Ci değerimiz 1 olacak. Kırmızı ile altı çizili olan şapkalı Ci değerimiz 1’e yakın olacak.

Bu 3 numaralı kısım hücrede nesne varsa kullanılıyor.

4 numaralı kısım da nesne yoksa kullanılıyor.

![Alt text](https://miro.medium.com/max/292/1*vGPfOabND6T_pulUTQD9gQ.png "img29")

![Alt text](https://miro.medium.com/max/324/1*36Cp6nPwnXYx588HEKmDYw.png "img30")

Denklemimizin beş numaralı kısmına bakarsak:

![Alt text](https://miro.medium.com/max/297/1*xEj6jK4se8HwAcR8A_YF_A.png "img31")

Bu kısım sınıflarımız için.

Hücrede, bounding boxta hangi obje var bulmak için.

Önce obje var mı diye bakıyor sonra 20 tane sınıftan hangi obje olduğunu buluyor.

Bu kısım için regression yöntemini kullandık.

Eğittiğimiz modelimizin bazı çıktıları:

![Alt text](https://miro.medium.com/max/272/1*EQgipGK9TbSohZsVu4rzrQ.png "img32")

![Alt text](https://miro.medium.com/max/273/1*wxCsiqaGNk3lkGm_jIEjNw.png "img33")

![Alt text](https://miro.medium.com/max/273/1*xpU14r2L5einwnwavPu0Uw.png "img34")

Biraz daha özelleştirilerek her sınıfa farklı renkte bounding box atanabilir ve her bir bounding box üzerinde nesnenin hangi sınıftan olduğu yazılabilir.


Kod üzerinde yapılan açıklamalarla kodun daha iyi anlaşılması amaçlanmaktadır.












