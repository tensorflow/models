"""
Used to plot the accuracy of the policy and value networks in
predicting professional game moves and results over the course
of training. Check FLAGS for default values for what models to
load and what sgf files to parse.

Usage:
python training_curve.py

Sample 3 positions from each game
python training_curve.py --num_positions=3

Only grab games after 2005 (default is 2000)
python training_curve.py --min_year=2005
"""
import sys; sys.path.insert(0, '.')
import sgf_wrapper

import go
import dual_net
import preprocessing
import selfplay_mcts
import evaluation
import sgf_wrapper
import utils
import rl_loop
import os
import shipname
import main
import numpy as np
import itertools
import matplotlib.pyplot as plt
import coords
import pandas as pd
import tensorflow as tf
import sgf
import pdb

from tqdm import tqdm
from sgf_wrapper import sgf_prop
from gtp_wrapper import make_gtp_instance, MCTSPlayer
from utils import logged_timer as timer
from tensorflow import gfile

tf.app.flags.DEFINE_string("sgf_dir", "sgf/baduk_db/", "sgf database")

tf.app.flags.DEFINE_string("model_dir", "saved_models",
                           "Where the model files are saved")
tf.app.flags.DEFINE_string("plot_dir", "data", "Where to save the plots.")
tf.app.flags.DEFINE_integer("min_year", "2000",
                            "Only take sgf games with date >= min_year")
tf.app.flags.DEFINE_string("komi", "7.5",
                            "Only take sgf games with given komi")
tf.app.flags.DEFINE_integer("idx_start", 150,
                            "Only take models after given idx")
tf.app.flags.DEFINE_integer("num_positions", 1,
                            "How many positions from each game to sample from.")
tf.app.flags.DEFINE_integer("eval_every", 5,
                            "Eval every k models to generate the curve")

FLAGS = tf.app.flags.FLAGS

def get_model_paths(model_dir):
    '''Returns all model paths in the model_dir.'''
    all_models = gfile.Glob(os.path.join(model_dir, '*.meta'))
    model_filenames = [os.path.basename(m) for m in all_models]
    model_numbers_names = [
        (shipname.detect_model_num(m), shipname.detect_model_name(m))
        for m in model_filenames]
    model_names = sorted(model_numbers_names)
    return [os.path.join(model_dir, name[1]) for name in model_names]

def load_player(model_path):
  print("Loading weights from %s ... " % model_path)
  with timer("Loading weights from %s ... " % model_path):
      network = dual_net.DualNetwork(model_path)
      network.name = os.path.basename(model_path)
  player = MCTSPlayer(network, verbosity=2)
  return player


def restore_params(model_path, player):
  with player.network.sess.graph.as_default():
    player.network.initialize_weights(model_path)

def batch_run_many(player, positions, batch_size=100):
  """Used to avoid a memory oveflow issue when running the network
  on too many positions. TODO: This should be a member function of
  player.network?"""
  prob_list = []
  value_list = []
  for idx in range(0, len(positions), batch_size):
    probs, values = player.network.run_many(positions[idx:idx+batch_size])
    prob_list.append(probs)
    value_list.append(values)
  return np.concatenate(prob_list, axis=0), np.concatenate(value_list, axis=0)


def eval_player(player, positions, moves, results):
  probs, values = batch_run_many(player, positions)
  policy_moves = [coords.from_flat(c) for c in np.argmax(probs, axis=1)]
  top_move_agree = [moves[idx] == policy_moves[idx] for idx in range(len(moves))]
  square_err = (values - results)**2/4
  return top_move_agree, square_err

def get_sgf_props(sgf_path):
  with open(sgf_path) as f:
    sgf_contents = f.read()
  collection = sgf.parse(sgf_contents)
  game = collection.children[0]
  props = game.root.properties
  return props

def parse_sgf(sgf_path):
  with open(sgf_path) as f:
    sgf_contents = f.read()

  collection = sgf.parse(sgf_contents)
  game = collection.children[0]
  props = game.root.properties
  assert int(sgf_prop(props.get('GM', ['1']))) == 1, "Not a Go SGF!"

  result = utils.parse_game_result(sgf_prop(props.get('RE')))

  positions, moves = zip(*[(p.position, p.next_move) for p in sgf_wrapper.replay_sgf(sgf_contents)])
  return positions, moves, result, props

def check_year(props, year):
  if year is None:
    return True
  if props.get('DT') is None:
    return False

  try:
    #Most sgf files in this database have dates of the form
    #"2005-01-15", but there are some rare exceptions like
    #"Broadcasted on 2005-01-15.
    year_sgf = int(props.get('DT')[0][:4])
  except:
    return False
  #TODO: better to use datetime comparison here?
  return year_sgf >= year

def check_komi(props, komi_str):
  if komi_str is None:
    return True
  if props.get('KM') is None:
    return False
  return props.get('KM')[0] == komi_str

def find_and_filter_sgf_files(base_dir, min_year = None, komi = None):
  sgf_files = []
  count = 0
  print("Finding all sgf files in {} with year >= {} and komi = {}".format(base_dir, min_year, komi))
  for i, (dirpath, dirnames, filenames) in enumerate(tqdm(os.walk(base_dir))):
    for filename in filenames:
      count += 1
      if count % 5000 == 0:
        print("Parsed {}, Found {}".format(count, len(sgf_files)))
      if filename.endswith('.sgf'):
        path = os.sep.join([dirpath, filename])
        props = get_sgf_props(path)
        if check_year(props, min_year) and check_komi(props, komi):
          sgf_files.append(path)
  print("Found {} sgf files matching filters".format(len(sgf_files)))
  return sgf_files

def sample_positions_from_games(sgf_files, num_positions=1):
  pos_data = []
  move_data = []
  result_data = []
  move_idxs = []

  fail_count = 0
  for i, path in enumerate(tqdm(sgf_files, desc="loading sgfs", unit="games")):
    try:
      positions, moves, result, props = parse_sgf(path)
    except KeyboardInterrupt:
      raise
    except:
      fail_count += 1
      continue

    #add entire game
    if num_positions== -1:
      pos_data.extend(positions)
      move_data.extend(moves)
      move_idxs.extend(range(len(positions)))
      result_data.extend([result for i in range(len(positions))])
    else:
      for idx in np.random.choice(len(positions), num_positions):
        pos_data.append(positions[idx])
        move_data.append(moves[idx])
        result_data.append(result)
        move_idxs.append(idx)
  print("Sampled {} positions, failed to parse {} files".format(len(pos_data), fail_count))
  return pos_data, move_data, result_data, move_idxs


def get_training_curve_data(model_dir, pos_data, move_data, result_data, idx_start=150, eval_every=10):
  model_paths = get_model_paths(model_dir)
  df = pd.DataFrame()
  player=None

  print("Evaluating models {}-{}, eval_every={}".format(idx_start, len(model_paths), eval_every))
  for idx in tqdm(range(idx_start, len(model_paths), eval_every)):
    if player:
      restore_params(model_paths[idx], player)
    else:
      player = load_player(model_paths[idx])

    correct, squared_errors = eval_player(player=player, positions=pos_data, moves=move_data, results=result_data)

    avg_acc = np.mean(correct)
    avg_mse = np.mean(squared_errors)
    print("Model: {}, acc: {:.4f}, mse: {:.4f}".format(model_paths[idx], avg_acc, avg_mse))
    df = df.append({"num": idx, "acc": avg_acc, "mse": avg_mse}, ignore_index=True)
  return df


def save_plots(data_dir, df):
  plt.plot(df["num"], df["acc"])
  plt.xlabel("Model idx")
  plt.ylabel("Accuracy")
  plt.title("Accuracy in Predicting Professional Moves")
  plot_path = os.sep.join([data_dir, "move_acc.pdf"])
  plt.savefig(plot_path)

  plt.figure()

  plt.plot(df["num"], df["mse"])
  plt.xlabel("Model idx")
  plt.ylabel("MSE/4")
  plt.title("MSE in predicting outcome")
  plot_path = os.sep.join([data_dir, "value_mse.pdf"])
  plt.savefig(plot_path)

def main(unusedargv):
  sgf_files = find_and_filter_sgf_files(FLAGS.sgf_dir, FLAGS.min_year, FLAGS.komi)
  pos_data, move_data, result_data, move_idxs = sample_positions_from_games(sgf_files=sgf_files, num_positions=FLAGS.num_positions)
  df = get_training_curve_data(FLAGS.model_dir, pos_data, move_data, result_data, FLAGS.idx_start, eval_every=FLAGS.eval_every)
  save_plots(FLAGS.plot_dir, df)

FLAGS = tf.app.flags.FLAGS

if __name__ == "__main__":
  tf.app.run(main)
