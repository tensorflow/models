(function($) {
$(document).ready(function() {
  $('.has-sub-menu > a span.mark').on('click', function(e) {
    $(this).parent().siblings('ul').slideToggle('fast', 'swing', function() {
      let text = '';
      if ($(this).is(':visible')) {
        text = '-';
      } else {
        text = '+';
      }
      $(this).siblings('a').children('span.mark').text(text);
    });
    e.preventDefault();
  });
});
})(jQuery);
