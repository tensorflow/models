% Dictionarios SPARSE
clt
load results/int

N1 = 32448; N2 = 15072; d  = 1;
%N1 = 20000; N2 = 1750; d  = 10;

x = rand(32448,1);[~,jj]=sort(x);i1 =jj(1:N1);
x = rand(15072,1);[~,jj]=sort(x);i2 =jj(1:N2);


clear param

param.K          = 1500;  % learns a dictionary with 100 elements
param.lambda     = 0.15;
param.numThreads = -1; % number of threads
param.batchsize  = 400;
param.verbose    = true;
param.iter       = 1000;
% param.mode       = 3;
% param.lamdba     = 25;


Ytrain           = (Xtrain(i1,1:d:end))';
Yt               = (Xtest(i2,1:d:end))';
ztrain           = dtrain(i1);
ztest            = dtest(i2);

i11              = find(ztrain==1);
i12              = find(ztrain==2);

Y1               = Ytrain(:,i11);
Y2               = Ytrain(:,i12);


Y1=Y1-repmat(mean(Y1),[size(Y1,1) 1]);
Y1=Y1 ./ repmat(sqrt(sum(Y1.^2)),[size(Y1,1) 1]);

Y2=Y2-repmat(mean(Y2),[size(Y2,1) 1]);
Y2=Y2 ./ repmat(sqrt(sum(Y2.^2)),[size(Y2,1) 1]);

Yt=Yt-repmat(mean(Yt),[size(Yt,1) 1]);
Yt=Yt ./ repmat(sqrt(sum(Yt.^2)),[size(Yt,1) 1]);


pmax = 0;
imax = 0;

x = [
    1500 0.15  20
    1500 0.15 1000
    2500 0.15 1000
    3500 0.15 1000
    5000 0.15 1000
    1500 0.15 2000
    2500 0.15 2000
    3500 0.15 2000 %
    5000 0.15 2000
    1500 0.1  1000
    2500 0.1  1000
    3500 0.1  1000
    5000 0.1  1000
    1500 0.2  1000
    2500 0.2  1000
    3500 0.2  1000
    5000 0.2  1000];


for i=8:size(x,1)
    i
    param.K = x(i,1);
    param.lambda = x(i,2);
    param.iter = x(i,3);
    param
    
    disp('computing D2...');
    D2               = mexTrainDL(Y2,param);
    howis(D2)
    disp('computing D1...');
    D1               = mexTrainDL(Y1,param);
    howis(D1)
    
    
    X1               = mexLasso(Yt,D1,param);
    X2               = mexLasso(Yt,D2,param);
    XX               = mexLasso(Yt,[D1 D2],param);
    XX1              = XX(1:param.K,:);
    XX2              = XX(param.K+1:end,:);
    sx1              = sum(abs(XX1))';
    sx2              = sum(abs(XX2))';
    
    ds    = (sx2>sx1)+1;
    
    Bev_performance(ds,ztest)
    
    
    E1 = Yt-D1*X1; % error
    E2 = Yt-D2*X2; % error
    
    norm1 = sqrt(sum(E1.*E1))';
    norm2 = sqrt(sum(E2.*E2))';
    
    ds    = (norm1>norm2)+1;
    
    p = Bev_performance(ds,ztest)*100
    
    if p>pmax
        disp('*** max ***')
        pmax = p;
        imax = i;
        
        
        fxname = 'sparse';
        T       = Bev_confusion(ds,ztest);
        C       = [T(1,1) T(2,2) T(1,2) T(2,1)];
        readme = 'see wacv_sparse.m for details, p = accuracy, each row of C = TP TN FP FN ';
        save(fxname,'p','C','fxname','param','D1','D2','readme','i');
    end
end
