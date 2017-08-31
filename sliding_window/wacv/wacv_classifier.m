function options = orl_cl(Xtrain,dtrain,opcl)

clname = opcl.clname;

options = Bcl_exe(clname,Xtrain,dtrain,opcl);


