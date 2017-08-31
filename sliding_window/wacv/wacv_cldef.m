function opcl = wacv_cldef(classifier,parameter)

switch lower(classifier)
    case 'knn'
        opcl.clname   = 'knn';
        opcl.k        = parameter;
    case 'libsvm'
        opcl.clname   = 'libsvm';
        opcl.kernel   = parameter;
    case 'ann'
        opcl.clname   = 'ann';
        opcl.hidden   = parameter;
    case 'src'
        opcl.clname   = 'src';
        opcl.T        = parameter;
    case 'sparse'
        opcl.clname   = 'sparse';
        opcl.K        = parameter(1); % number of atoms of the dictionary
        opcl.L        = parameter(2); % sparsity (number of atoms used for the representation)
        opcl.iternum  = 30;
        opcl.type     = 1;
end
