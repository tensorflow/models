info = wacv_demo('clp','knn',3);           p0 = info.acc;
info = wacv_demo('vgg4','knn',3);          p1 = info.acc;
info = wacv_demo('vgg3','knn',3);          p2 = info.acc;
info = wacv_demo('vgg1','knn',5);          p3 = info.acc;
info = wacv_demo('alx','knn',3);           p4 = info.acc;
info = wacv_demo('slbp','libsvm','-t -0'); p5 = info.acc;
