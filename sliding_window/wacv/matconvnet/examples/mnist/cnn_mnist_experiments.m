%% Experiment with the cnn_mnist_fc_bnorm

[net_bn, info_bn] = cnn_mnist(...
  'expDir', 'data/mnist-bnorm', 'batchNormalization', true);

[net_fc, info_fc] = cnn_mnist(...
  'expDir', 'data/mnist-baseline', 'batchNormalization', false);

figure(1) ; clf ;
subplot(1,2,1) ;
semilogy([info_fc.val.objective]', 'o-') ; hold all ;
semilogy([info_bn.val.objective]', '+--') ;
xlabel('Training samples [x 10^3]'); ylabel('energy') ;
grid on ;
h=legend('BSLN', 'BNORM') ;
set(h,'color','none');
title('objective') ;
subplot(1,2,2) ;
plot([info_fc.val.top1err]', 'o-') ; hold all ;
plot([info_fc.val.top5err]', '*-') ;
plot([info_bn.val.top1err]', '+--') ;
plot([info_bn.val.top5err]', 'x--') ;
h=legend('BSLN-val','BSLN-val-5','BNORM-val','BNORM-val-5') ;
grid on ;
xlabel('Training samples [x 10^3]'); ylabel('error') ;
set(h,'color','none') ;
title('error') ;
drawnow ;
