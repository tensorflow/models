(function() {
  let modules = {
    262: function() {
      function t(t) {
        t.currentTarget.setAttribute('class', 'copy-btn copy');
        t.currentTarget.setAttribute('aria-label', 'Copy this code.');
      }
      window.onload = function() {
        new ClipboardJS('.copy', {
          target: function(t) {
            return t.nextElementSibling;
          }
        })
        .on('success', function(t) {
          t.trigger.setAttribute('class', 'copy-btn copy tooltipped');
          t.trigger.setAttribute('aria-label', 'Copied!');
          t.clearSelection();
        })
        .on('error', function(t) {
          console.error('Action:', t.action);
          console.error('Trigger:', t.trigger);
        });
      };
      document.addEventListener('DOMContentLoaded', function() {
        document.querySelectorAll('.copy').forEach(function(o) {
          o.addEventListener('animationend', t);
        });
      });
    },
    169: function() {
      document.addEventListener('DOMContentLoaded', function() {
        document.querySelector('main')
          .querySelectorAll('h1, h2, h3, h4, h5, h6')
          .forEach(function(t) {
            if (t.id) {
              let o = document.createElement('a');
              o.setAttribute('class', 'headerlink');
              o.setAttribute('href', '#' + t.id);
              o.setAttribute('title', 'Permalink to this headline');
              let e = document.createElement('i');
              e.setAttribute('class', 'fas fa-hashtag');
              e.setAttribute('aria-hidden', 'true');
              o.append(e);
              t.append(o);
            }
          });
      });
    },
    337: function() {
      (function($) {
        $(window).on('load.BackToTheTop', function() {
          $('a[href^="#"]').BackToTheTop();
        });
        $.fn.BackToTheTop = function(options) {
          let settings = $.extend({
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
            position: 'fixed'
          }, options);
          
          function scrollToTarget() {
            $('a[href^="#"]').on('click.BackToTheTop', function(event) {
              let targetOffset = $(this).data('backtothetop-scrolltop') !== undefined
                ? $(this).data('backtothetop-scrolltop')
                : $(this.hash).offset() !== undefined ? $(this.hash).offset().top : null;

              if (targetOffset !== null) {
                let duration = $(this).data('backtothetop-duration') !== undefined
                  ? $(this).data('backtothetop-duration')
                  : settings.duration;

                let easing = $(this).data('backtothetop-easing') !== undefined
                  ? $(this).data('backtothetop-easing')
                  : settings.easing;

                let offset = $(this).data('backtothetop-offset') !== undefined
                  ? $(this).data('backtothetop-offset')
                  : settings.offset;

                let hash = $(this).data('backtothetop-hash') !== undefined
                  ? $(this).data('backtothetop-hash')
                  : settings.hash;

                let href = $(this).attr('href');

                $('html, body').animate({ scrollTop: targetOffset + offset }, duration, easing, function() {
                  if (hash) {
                    window.history.pushState('', '', href);
                  }
                });
                event.preventDefault();
              }
            });
          }

          function handleFixedButton() {
            let fixedButton = $('a#backtothetop-fixed');
            if (fixedButton.length) {
              let scrollOffset = fixedButton.data('backtothetop-fixed-scroll-offset') !== undefined
                ? fixedButton.data('backtothetop-fixed-scroll-offset')
                : settings.scrolloffset;

              let fadeIn = fixedButton.data('backtothetop-fixed-fadein') !== undefined
                ? fixedButton.data('backtothetop-fixed-fadein')
                : settings.fadein;

              let fadeOut = fixedButton.data('backtothetop-fixed-fadeout') !== undefined
                ? fixedButton.data('backtothetop-fixed-fadeout')
                : settings.fadeout;

              let display = fixedButton.data('backtothetop-fixed-display') !== undefined
                ? fixedButton.data('backtothetop-fixed-display')
                : settings.display;

              let top = fixedButton.data('backtothetop-fixed-top') !== undefined
                ? fixedButton.data('backtothetop-fixed-top')
                : settings.top;

              let bottom = fixedButton.data('backtothetop-fixed-bottom') !== undefined
                ? fixedButton.data('backtothetop-fixed-bottom')
                : settings.bottom;

              let left = fixedButton.data('backtothetop-fixed-left') !== undefined
                ? fixedButton.data('backtothetop-fixed-left')
                : settings.left;

              let right = fixedButton.data('backtothetop-fixed-right') !== undefined
                ? fixedButton.data('backtothetop-fixed-right')
                : settings.right;

              let zIndex = fixedButton.data('backtothetop-fixed-zindex') !== undefined
                ? fixedButton.data('backtothetop-fixed-zindex')
                : settings.zIndex;

              if (display === 'top-left') {
                bottom = 'none';
                right = 'none';
              } else if (display === 'top-right') {
                bottom = 'none';
                left = 'none';
              } else if (display === 'bottom-left') {
                top = 'none';
                right = 'none';
              } else if (display === 'bottom-right') {
                top = 'none';
                left = 'none';
              }

              fixedButton.css({ display: 'none' });

              $(window).on('scroll.BackToTheTop', function() {
                if ($(this).scrollTop() > scrollOffset) {
                  fixedButton.css({
                    top: top,
                    bottom: bottom,
                    left: left,
                    right: right,
                    zIndex: zIndex,
                    position: settings.position
                  });

                  if (fixedButton.css('display') === 'none') {
                    fixedButton.fadeIn(fadeIn);
                  }
                } else if ($(this).scrollTop() <= 0 + scrollOffset && fixedButton.css('display') !== 'none') {
                  fixedButton.fadeOut(fadeOut);
                }
              });
            }
          }

          scrollToTarget();
          handleFixedButton();

          return this;
        };
      })(jQuery);
    },
    670: function() {
      document.addEventListener('DOMContentLoaded', function() {
        let prev = document.querySelector('.nav-prev');
        let next = document.querySelector('.nav-next');
        document.addEventListener('keydown', function(e) {
          if (prev && e.key === 'ArrowLeft') {
            location.href = prev.getAttribute('href');
          }
          if (next && e.key === 'ArrowRight') {
            location.href = next.getAttribute('href');
          }
        });
      });
    },
    598: function() {
      (function($) {
        $(document).ready(function() {
          $('.has-sub-menu > a span.mark').on('click', function(e) {
            $(this).parent().siblings('ul').slideToggle('fast', 'swing', function() {
              let mark;
              mark = $(this).is(':visible') ? '-' : '+';
              $(this).siblings('a').children('span.mark').text(mark);
            });
            e.preventDefault();
          });
        });
      })(jQuery);
    }
  };
  
  let executedModules = {};

  function require(moduleId) {
    if (executedModules[moduleId] !== undefined) {
      return executedModules[moduleId].exports;
    }
    let module = executedModules[moduleId] = {
      exports: {}
    };
    modules[moduleId](module, module.exports, require);
    return module.exports;
  }

  require(598);
  require(670);
  require(337);
  require(169);
  require(262);
})();