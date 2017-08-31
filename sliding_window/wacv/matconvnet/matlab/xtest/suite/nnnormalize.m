classdef nnnormalize < nntest
  properties (TestParameter)
    group = {2 3 4 5 6 8 9 10 11 12 13 14 15 16 17}
    sgroup = {2 3 4 5 6 7}
  end

  methods (Test)
    function basic(test, group)
      param = [group, .1, .5, .75] ;
      x = test.randn(3,2,10,4) ;
      y = vl_nnnormalize(x,param) ;
      dzdy = test.rand(size(y))-0.5 ;
      dzdx = vl_nnnormalize(x,param,dzdy) ;
      test.der(@(x) vl_nnnormalize(x,param), x, dzdy, dzdx, test.range * 1e-3, 0.3) ;
    end

    function compare_to_naive(test, sgroup)
      param = [sgroup, .1, .5, .75] ;
      x = test.randn(3,2,10,4) ;
      y = vl_nnnormalize(gather(x),param) ;
      y_ = test.zeros(size(y)) ;
      x_ = gather(x) ;
      for i=1:size(x,1)
        for j=1:size(x,2)
          for n=1:size(x,4)
            t = test.zeros(1,1,size(x,3),1) ;
            t(1,1,:,1) = (param(2) + param(3)*conv(squeeze(x_(i,j,:,n)).^2, ...
                                                   ones(param(1),1), 'same')).^(-param(4)) ;
            y_(i,j,:,n) = x_(i,j,:,n) .* t ;
          end
        end
      end
      test.eq(y,y_) ;
    end

    function l2(test)
      x = test.randn(1,1,10,1) ;
      y = vl_nnnormalize(x, [20, 0, 1, .5]) ;
      test.eq(sum(y(:).^2), test.toDataType(1), 1e-2) ;
    end
  end
end
