(function() {
window.onload =
    function() {
  new ClipboardJS('.copy', {
    target: function(trigger) {
      return trigger.nextElementSibling;
    }
  })
      .on('success',
          function(e) {
            showTooltip(e.trigger, 'Copied!');
            e.clearSelection();
          })
      .on('error', function(e) {
        console.error('Action:', e.action);
        console.error('Trigger:', e.trigger);
      });
};

    document.addEventListener('DOMContentLoaded', function() {
      const btns = document.querySelectorAll('.copy');

      btns.forEach((el) => {
        el.addEventListener('animationend', clearTooltip);
      });
    });

function showTooltip(e, msg) {
  e.setAttribute('class', 'copy-btn copy tooltipped');
  e.setAttribute('aria-label', msg);
}

function clearTooltip(e) {
  e.currentTarget.setAttribute('class', 'copy-btn copy');
  e.currentTarget.setAttribute('aria-label', 'Copy this code.');
}
})();
