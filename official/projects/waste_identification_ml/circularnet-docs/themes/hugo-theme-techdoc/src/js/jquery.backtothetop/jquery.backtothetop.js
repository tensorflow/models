(function($) {
  $(window).on('load.BackToTheTop', function() {
    $('a[href^="#"]').BackToTheTop();
  });

  $.fn.BackToTheTop = function( options ) {

    let defaults = {
      duration: 300,
      easing: 'swing',
      offset: 0,
      hash: true,
      scrolloffset: 0,
      fadein: 'slow',
      fadeout: 'slow',
      display: 'bottom-right',
      top: 0,
      bottom: 0,
      left: 0,
      right: 0,
      zIndex: 999,
      position : 'fixed'
    };

    $.extend( defaults, options );

    let init = function() {
      $('a[href^="#"]').on('click.BackToTheTop', function() {
        let scrollTop =
            $(this).data('backtothetop-scrolltop') !== undefined ? $(this).data('backtothetop-scrolltop')
          : $(this.hash).offset() ? $(this.hash).offset().top
          : $(this).attr('id') == 'backtothetop-fixed' && $(this).attr('href') == '#' ? 0
          : null ;

        if (scrollTop === null)
          return;

        let duration = typeof $(this).data('backtothetop-duration') === "undefined" ? defaults.duration : $(this).data('backtothetop-duration');
        let easing = typeof $(this).data('backtothetop-easing') === "undefined" ?  defaults.easing : $(this).data('backtothetop-easing');
        let offset = typeof $(this).data('backtothetop-offset') === "undefined" ? defaults.offset : $(this).data('backtothetop-offset');
        let hash = typeof $(this).data('backtothetop-hash') === "undefined" ? defaults.hash : $(this).data('backtothetop-hash');
        let href = $(this).attr('href');

        $('html,body').animate(
          { 'scrollTop' : scrollTop + offset }, duration, easing,
          function() {
            if (hash === true) {
              window.history.pushState('', '', href);
            }
          }
        );

        return false;
      });
    };

    let fixed = function() {
      let elem = $('a#backtothetop-fixed');
      if ( !elem )
        return;
      let scrollOffset = typeof elem.data('backtothetop-fixed-scroll-offset') === "undefined" ? defaults.scrolloffset : elem.data('backtothetop-fixed-scroll-offset');
      let fadeIn = typeof elem.data('backtothetop-fixed-fadein') === "undefined" ? defaults.fadein : elem.data('backtothetop-fixed-fadein');
      let fadeOut = typeof elem.data('backtothetop-fixed-fadeout') === "undefined" ? defaults.fadeout : elem.data('backtothetop-fixed-fadeout');
      let display = typeof elem.data('backtothetop-fixed-display') === "undefined" ? defaults.display : elem.data('backtothetop-fixed-display');
      let top = typeof elem.data('backtothetop-fixed-top') === "undefined" ? defaults.top : elem.data('backtothetop-fixed-top');
      let bottom = typeof elem.data('backtothetop-fixed-bottom') === "undefined" ? defaults.bottom : elem.data('backtothetop-fixed-bottom');
      let left = typeof elem.data('backtothetop-fixed-left') === "undefined" ? defaults.left : elem.data('backtothetop-fixed-left');
      let right = typeof elem.data('backtothetop-fixed-right') === "undefined" ? defaults.right : elem.data('backtothetop-fixed-right');
      let zindex = typeof elem.data('backtothetop-fixed-zindex') === "undefined" ? defaults.zIndex : elem.data('backtothetop-fixed-zindex');

      if (display == 'top-left') {
        bottom = 'none';
        right = 'none';
      }
      else if (display == 'top-right') {
        bottom = 'none';
        left = 'none';
      }
      else if (display == 'bottom-left') {
        top = 'none';
        right = 'none';
      }
      else if (display == 'bottom-right') {
        top = 'none';
        left = 'none';
      }

      elem.css({ 'display' : 'none' });

      $(window).on('scroll.BackToTheTop', function () {
        if ($(this).scrollTop() > scrollOffset) {
          elem.css({
            'top' : top,
            'bottom' : bottom,
            'left' : left,
            'right' : right,
            'zIndex' : zindex,
            'position' : defaults.position
          });

          if (elem.css('display') == 'none' ) {
            elem.fadeIn(fadeIn);
          }

        }
        else if ($(this).scrollTop() <= 0 + scrollOffset) {
          if (elem.css('display') != 'none' ) {
            elem.fadeOut(fadeOut);
          }
        }
      });
    };

    init();
    fixed();
  };
})(jQuery);