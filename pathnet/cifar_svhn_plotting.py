import random,os
import numpy as np
from matplotlib import pyplot
result_folder1="pathnet2";
result_filename1="cifar_svhn.log";
result_folder2="pathnet2";
result_filename2="svhn_cifar.log";
figure_filename="cifar_svhn.PNG";

#First Results
f=open(os.path.join(result_folder1,result_filename1),"r");
data1=np.array([i.split(",")[2:4] for i in f.readlines()]);
for i in range(len(data1)):
  for j in range(2):
    data1[i][j]=int(float(data1[i][j].split(":")[1])*100.0);
#Second Results
f=open(os.path.join(result_folder2,result_filename2),"r");
data2=np.array([i.split(",")[2:4] for i in f.readlines()]);
for i in range(len(data2)):
  for j in range(2):
    data2[i][j]=int(float(data2[i][j].split(":")[1])*100.0);

bins = np.linspace(0, 70, 35);

##Save Figure
#First Subplot
x1=np.array(data1[:,0],dtype=int);
x2=np.array(data2[:,1],dtype=int);
fig=pyplot.figure(figsize=(20,5));
sub1=fig.add_subplot(1,2,1);
sub1.hist(x1, bins, alpha=0.5, label='first')
sub1.hist(x2, bins, alpha=0.5, label='second('+figure_filename.split(".")[0].split("_")[1]+"->"+figure_filename.split(".")[0].split("_")[0]+')')
sub1.set_title(figure_filename.split(".")[0].split("_")[0]);
sub1.set_xlabel("accuracy");
sub1.set_ylabel("frequencies");
sub1.legend(loc='upper right')
#Second Subplot
x1=np.array(data2[:,0],dtype=int);
x2=np.array(data1[:,1],dtype=int);
sub2=fig.add_subplot(1,2,2);
sub2.hist(x1, bins, alpha=0.5, label='first')
sub2.hist(x2, bins, alpha=0.5, label='second('+figure_filename.split(".")[0].split("_")[0]+"->"+figure_filename.split(".")[0].split("_")[1]+')')
sub2.set_title(figure_filename.split(".")[0].split("_")[1]);
sub2.set_xlabel("accuracy");
sub2.set_ylabel("frequencies");
sub2.legend(loc='upper right')

pyplot.savefig(os.path.join("pathnet","figures",figure_filename));

