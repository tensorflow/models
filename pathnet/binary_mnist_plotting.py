import random,os
import numpy as np
from matplotlib import pyplot
result_folder1="pathnet3";
result_filename1="binary_mnist_pathnet.res";
result_folder2="pathnet4";
result_filename2="binary_mnist_pathnet.res";
figure_filename="6vs7_4vs5.PNG";

#First Results
f=open(os.path.join(result_folder1,result_filename1),"r");
data1=np.array([i.split(",")[2:4] for i in f.readlines()]);
for i in range(len(data1)):
  for j in range(2):
    data1[i][j]=int(data1[i][j].split(":")[1]);
#Second Results
f=open(os.path.join(result_folder2,result_filename2),"r");
data2=np.array([i.split(",")[2:4] for i in f.readlines()]);
for i in range(len(data2)):
  for j in range(2):
    data2[i][j]=int(data2[i][j].split(":")[1]);

bins = np.linspace(0, 300, 30)

##Save Figure
#First Subplot
x1=np.array(data1[:,0],dtype=int);
x2=np.array(data2[:,1],dtype=int);
fig=pyplot.figure(figsize=(20,10));
sub1=fig.add_subplot(1,2,1);
sub1.hist(x1, bins, alpha=0.5, label='first')
sub1.hist(x2, bins, alpha=0.5, label='second('+figure_filename.split(".")[0].split("_")[1]+"->"+figure_filename.split(".")[0].split("_")[0]+')')
sub1.set_title(figure_filename.split(".")[0].split("_")[0]);
sub1.set_xlabel("generations");
sub1.set_ylabel("frequencies");
sub1.legend(loc='upper right')
#Second Subplot
x1=np.array(data2[:,0],dtype=int);
x2=np.array(data1[:,1],dtype=int);
sub2=fig.add_subplot(1,2,2);
sub2.hist(x1, bins, alpha=0.5, label='first')
sub2.hist(x2, bins, alpha=0.5, label='second('+figure_filename.split(".")[0].split("_")[0]+"->"+figure_filename.split(".")[0].split("_")[1]+')')
sub2.set_title(figure_filename.split(".")[0].split("_")[1]);
sub2.set_xlabel("generations");
sub2.set_ylabel("frequencies");
sub2.legend(loc='upper right')

pyplot.savefig(os.path.join("pathnet","figures","binary_mnist_"+figure_filename));

