from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import matplotlib
import os

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

plt.style.use('ggplot')

FLAGS = flags.FLAGS
flags.DEFINE_string('plot_file', '', 'Output file name.')

qa_lnmax = [500, 750] + range(1000, 12500, 500)

acc_lnmax = [43.3, 52.3, 59.8, 66.7, 68.8, 70.5, 71.6, 72.3, 72.6, 72.9, 73.4,
             73.4, 73.7, 73.9, 74.2, 74.4, 74.5, 74.7, 74.8, 75, 75.1, 75.1,
             75.4, 75.4, 75.4]

qa_gnmax = [456, 683, 908, 1353, 1818, 2260, 2702, 3153, 3602, 4055, 4511, 4964,
            5422, 5875, 6332, 6792, 7244, 7696, 8146, 8599, 9041, 9496, 9945,
            10390, 10842]

acc_gnmax = [39.6, 52.2, 59.6, 66.6, 69.6, 70.5, 71.8, 72, 72.7, 72.9, 73.3,
             73.4, 73.4, 73.8, 74, 74.2, 74.4, 74.5, 74.5, 74.7, 74.8, 75, 75.1,
             75.1, 75.4]

qa_gnmax_aggressive = [167, 258, 322, 485, 647, 800, 967, 1133, 1282, 1430,
                       1573, 1728, 1889, 2028, 2190, 2348, 2510, 2668, 2950,
                       3098, 3265, 3413, 3581, 3730]

acc_gnmax_aggressive = [17.8, 26.8, 39.3, 48, 55.7, 61, 62.8, 64.8, 65.4, 66.7,
                        66.2, 68.3, 68.3, 68.7, 69.1, 70, 70.2, 70.5, 70.9,
                        70.7, 71.3, 71.3, 71.3, 71.8]


def main(argv):
  del argv  # Unused.

  plt.close('all')
  fig, ax = plt.subplots()
  fig.set_figheight(4.7)
  fig.set_figwidth(5)
  ax.plot(qa_lnmax, acc_lnmax, color='r', ls='--', linewidth=5., marker='o',
          alpha=.5, label='LNMax')
  ax.plot(qa_gnmax, acc_gnmax, color='g', ls='-', linewidth=5., marker='o',
          alpha=.5, label='Confident-GNMax')
  # ax.plot(qa_gnmax_aggressive, acc_gnmax_aggressive, color='b', ls='-', marker='o', alpha=.5, label='Confident-GNMax (aggressive)')
  plt.xticks([0, 2000, 4000, 6000])
  plt.xlim([0, 6000])
  # ax.set_yscale('log')
  plt.ylim([65, 76])
  ax.tick_params(labelsize=14)
  plt.xlabel('Number of queries answered', fontsize=16)
  plt.ylabel('Student test accuracy (%)', fontsize=16)
  plt.legend(loc=2, prop={'size': 16})

  x = [400, 2116, 4600, 4680]
  y = [69.5, 68.5, 74, 72.5]
  annotations = [0.76, 2.89, 1.42, 5.76]
  color_annotations = ['g', 'r', 'g', 'r']
  for i, txt in enumerate(annotations):
    ax.annotate(r'${\varepsilon=}$' + str(txt), (x[i], y[i]), fontsize=16,
                color=color_annotations[i])

  plot_filename = os.path.expanduser(FLAGS.plot_file)
  plt.savefig(plot_filename, bbox_inches='tight')
  plt.show()

if __name__ == '__main__':
  app.run(main)
