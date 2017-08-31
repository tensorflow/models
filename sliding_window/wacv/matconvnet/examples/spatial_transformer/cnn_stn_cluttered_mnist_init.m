function nn = cnn_stn_cluttered_mnist_init(imsz, use_transformer)
% script to initialize a small spatial transformer networl
% for cluttered MNIST:

  % init the object:
  nn = dagnn.DagNN();

  if use_transformer
    % ************************** localization network ****************************
    l_mp1 = dagnn.Pooling('method', 'max', 'poolSize', [2 2],'pad', 0, 'stride', 2);
    nn.addLayer('l_mp1', l_mp1, {'input'}, {'x1'});
    l_cnv1 = dagnn.Conv('size',[5 5 1 20],'pad',0,'stride',1,'hasBias',true);
    nn.addLayer('l_cnv1', l_cnv1, {'x1'}, {'x2'}, {'lc1f','lc1b'});
    nn.addLayer('l_re1', dagnn.ReLU(), {'x2'}, {'x3'});

    l_mp2 = dagnn.Pooling('method', 'max', 'poolSize', [2 2],'pad', 0, 'stride', 2);
    nn.addLayer('l_mp2', l_mp2, {'x3'}, {'x4'});
    l_cnv2 = dagnn.Conv('size',[5 5 20 20],'pad',0,'stride',1,'hasBias',true);
    nn.addLayer('l_cnv2', l_cnv2, {'x4'}, {'x5'}, {'lc2f','lc2b'});
    nn.addLayer('l_re2', dagnn.ReLU(), {'x5'}, {'x6'});

    l_fc1 = dagnn.Conv('size',[9,9,20,50],'pad',0,'stride',1,'hasBias',true);
    nn.addLayer('l_fc1', l_fc1, {'x6'}, {'x7'}, {'lfcf','lfcb'});
    nn.addLayer('l_re3', dagnn.ReLU(), {'x7'}, {'x8'});

    % output affine transforms:
    l_out = dagnn.Conv('size',[1,1,50,6],'pad',0,'stride',1,'hasBias',true);
    nn.addLayer('l_out', l_out, {'x8'}, {'aff'}, {'lof','lob'});
    %***** NEED TO SET THE PARAMETERS OF THIS LAST LAYER TO OUTPUT IDENTITY *******

    % ************************** spatial transformer ******************************
    aff_grid = dagnn.AffineGridGenerator('Ho',imsz(1),'Wo',imsz(2));
    nn.addLayer('aff', aff_grid,{'aff'},{'grid'});

    sampler = dagnn.BilinearSampler();
    nn.addLayer('samp',sampler,{'input','grid'},{'xST'});
    % *****************************************************************************
  end

  % ************************** classification network ***************************
  % average pooling:
  c_ap1 = dagnn.Pooling('method', 'avg', 'poolSize', [2 2],'pad', 0, 'stride', 2);
  in_name = 'input';
  if use_transformer, in_name = 'xST'; end
  nn.addLayer('c_ap1', c_ap1, {in_name}, {'xc0'}); %output dim: 60/2 = 30

  % classification net:
  c_cnv1 = dagnn.Conv('size',[7 7 1 32],'pad',0,'stride',1,'hasBias',true);
  nn.addLayer('c_cnv1', c_cnv1, {'xc0'}, {'xc1'}, {'cc1f','cc1b'}); % output dim = 30-6 = 24
  nn.addLayer('c_re1', dagnn.ReLU(), {'xc1'}, {'xc2'});
  c_mp1 = dagnn.Pooling('method', 'max', 'poolSize', [2 2],'pad', 0, 'stride', 2);
  nn.addLayer('c_mp1', c_mp1, {'xc2'}, {'xc3'}); % output dim = 24/2 = 12

  c_cnv2 = dagnn.Conv('size',[5 5 32 48],'pad',0,'stride',1,'hasBias',true);
  nn.addLayer('c_cnv2', c_cnv2, {'xc3'}, {'xc4'}, {'cc2f','cc2b'}); %output dim = 12-4 = 8
  nn.addLayer('c_re2', dagnn.ReLU(), {'xc4'}, {'xc5'});
  c_mp2 = dagnn.Pooling('method', 'max', 'poolSize', [2 2],'pad', 0, 'stride', 2);
  nn.addLayer('c_mp2', c_mp2, {'xc5'}, {'xc6'}); %output dim = 8/2 = 4

  c_fc1 = dagnn.Conv('size',[4 4 48 256],'pad',0,'stride',1,'hasBias',true);
  nn.addLayer('c_fc1', c_fc1, {'xc6'}, {'xc7'}, {'cfcf','cfcb'});
  nn.addLayer('c_re3', dagnn.ReLU(), {'xc7'}, {'xc8'});

  c_fc2 = dagnn.Conv('size',[1 1 256 10],'pad',0,'stride',1,'hasBias',true);
  nn.addLayer('c_fc2', c_fc2, {'xc8'}, {'pred'}, {'cof','cob'});

  % softmax loss:
  nn.addLayer('loss', dagnn.Loss('loss', 'softmaxlog'), {'pred','label'}, 'objective');
  nn.addLayer('error', dagnn.Loss('loss', 'classerror'), {'pred','label'}, 'error');
  % ***************************************************************************** 
  % initialize the weights:
  nn.initParams();

  if use_transformer
    % VERY IMPORTANT: bias the transformation to IDENTITY:
    f_prev = nn.params(nn.getParamIndex('lof')).value;
    nn.params(nn.getParamIndex('lof')).value = 0*f_prev;

    b_prev = 0*nn.params(nn.getParamIndex('lob')).value;
    b_prev(1) = 1; b_prev(4) = 1;
    nn.params(nn.getParamIndex('lob')).value = b_prev;
  end

  nn.meta.trainOpts.learningRate = 0.001 ;
  nn.meta.trainOpts.batchSize = 256 ;
  nn.meta.trainOpts.numEpochs = 60 ;
end
