# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""Script to render object views from ShapeNet obj models.

Example usage:
  blender -b --python render.py -- -m model.obj -o output/ -s 128 -n 120 -fov 5

"""
from __future__ import print_function

import argparse
import itertools
import json
from math import pi
import os
import random
import sys
from mathutils import Vector
import math
import mathutils
import time
import copy

import bpy

sys.path.append(os.path.dirname(__file__))

BG_LUMINANCE = 0


def look_at(obj_camera, point):
  loc_camera = obj_camera.location
  direction = point - loc_camera
  # point the cameras '-Z' and use its 'Y' as up
  rot_quat = direction.to_track_quat('-Z', 'Y')

  obj_camera.rotation_euler = rot_quat.to_euler()


def roll_camera(obj_camera):
  roll_rotate = mathutils.Euler(
      (0, 0, random.random() * math.pi - math.pi * 0.5), 'XYZ')
  obj_camera.rotation_euler = (obj_camera.rotation_euler.to_matrix() *
      roll_rotate.to_matrix()).to_euler()


def norm(x):
  return math.sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2])


def normalize(x):
  n = norm(x)
  x[0] /= n
  x[1] /= n
  x[2] /= n


def random_top_sphere():
  xyz = [random.normalvariate(0, 1) for x in range(3)]
  normalize(xyz)

  if xyz[2] < 0:
    xyz[2] *= -1
  return xyz


def perturb_sphere(loc, size):
  while True:
    xyz = [random.normalvariate(0, 1) for x in range(3)]
    normalize(xyz)

    nloc = [loc[i] + xyz[i] * random.random() * size for i in range(3)]
    normalize(nloc)

    if nloc[2] >= 0:
      return nloc


def perturb(loc, size):
  while True:
    nloc = [loc[i] + random.random() * size * 2 - size for i in range(3)]
    if nloc[2] >= 0:
      return nloc

    bpy.ops.object.mode_set()


def delete_all_objects():
  bpy.ops.object.select_by_type(type="MESH")
  bpy.ops.object.delete(use_global=False)


def set_scene(render_size, fov, alpha=False):
  """Set up default scene properties."""
  delete_all_objects()

  cam = bpy.data.cameras["Camera"]
  cam.angle = fov * pi / 180

  light = bpy.data.objects["Lamp"]
  light.location = (0, 0, 1)
  look_at(light, Vector((0.0, 0, 0)))
  bpy.data.lamps['Lamp'].type = "HEMI"
  bpy.data.lamps['Lamp'].energy = 1
  bpy.data.lamps['Lamp'].use_specular = False
  bpy.data.lamps['Lamp'].use_diffuse = True

  bpy.context.scene.world.horizon_color = (
      BG_LUMINANCE, BG_LUMINANCE, BG_LUMINANCE)

  bpy.context.scene.render.resolution_x = render_size
  bpy.context.scene.render.resolution_y = render_size
  bpy.context.scene.render.resolution_percentage = 100

  bpy.context.scene.render.use_antialiasing = True
  bpy.context.scene.render.antialiasing_samples = '5'


def get_modelview_matrix():
  cam = bpy.data.objects["Camera"]
  bpy.context.scene.update()

  # when apply to object with CV coordinate i.e. to_blender * obj
  # this gives object in blender coordinate
  to_blender = mathutils.Matrix(
      ((1., 0., 0., 0.),
       (0., 0., -1., 0.),
       (0., 1., 0., 0.),
       (0., 0., 0., 1.)))
  return cam.matrix_world.inverted() * to_blender


def print_matrix(f, mat):
  for i in range(4):
    for j in range(4):
      f.write("%lf " % mat[i][j])
    f.write("\n")


def mul(loc, v):
  return [loc[i] * v for i in range(3)]


def merge_all():
  bpy.ops.object.select_by_type(type="MESH")
  bpy.context.scene.objects.active = bpy.context.selected_objects[0]
  bpy.ops.object.join()
  obj = bpy.context.scene.objects.active
  bpy.ops.object.origin_set(type="ORIGIN_CENTER_OF_MASS")
  return obj


def insert_frame(obj, frame_number):
  obj.keyframe_insert(data_path="location", frame=frame_number)
  obj.keyframe_insert(data_path="rotation_euler", frame=frame_number)
  obj.keyframe_insert(data_path="scale", frame=frame_number)


def render(output_prefix):
  bpy.context.scene.render.filepath = output_prefix
  bpy.context.scene.render.image_settings.file_format = "PNG"
  bpy.context.scene.render.alpha_mode = "TRANSPARENT"
  bpy.context.scene.render.image_settings.color_mode = "RGBA"
  bpy.ops.render.render(write_still=True, animation=True)


def render_obj(
    obj_fn, save_dir, n, perturb_size, rotate=False, roll=False, scale=1.0):

  # Load object.
  bpy.ops.import_scene.obj(filepath=obj_fn)
  cur_obj = merge_all()

  scale = 2.0 / max(cur_obj.dimensions) * scale
  cur_obj.scale = (scale, scale, scale)
  # Using the center of mass as the origin doesn't really work, because Blender
  # assumes the object is a solid shell. This seems to generate better-looking
  # rotations.

  bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')

  # bpy.ops.mesh.primitive_cube_add(location=(0, 0, 1))
  # cube = bpy.data.objects["Cube"]
  # cube.scale = (0.2, 0.2, 0.2)

  for polygon in cur_obj.data.polygons:
    polygon.use_smooth = True

  bpy.ops.object.select_all(action="DESELECT")

  camera = bpy.data.objects["Camera"]

  # os.system("mkdir " + save_dir)
  for i in range(n):
    fo = open(save_dir + "/%06d.txt" % i, "w")
    d = 30
    shift = 0.2
    if rotate:
      t = 1.0 * i / (n-1) * 2 * math.pi
      loc = [math.sin(t), math.cos(t), 1]

      normalize(loc)
      camera.location = mul(loc, d)
      look_at(camera, Vector((0.0, 0, 0)))

      print_matrix(fo, get_modelview_matrix())
      print_matrix(fo, get_modelview_matrix())

      insert_frame(camera, 2 * i)
      insert_frame(camera, 2 * i + 1)

    else:
      loc = random_top_sphere()

      camera.location = mul(loc, d)
      look_at(camera, Vector((0.0, 0, 0)))

      if roll:
        roll_camera(camera)
      camera.location = perturb(mul(loc, d), shift)

      print_matrix(fo, get_modelview_matrix())
      insert_frame(camera, 2 * i)

      if perturb_size > 0:
        loc = perturb_sphere(loc, perturb_size)
      else:
        loc = random_top_sphere()

      camera.location = mul(loc, d)
      look_at(camera, Vector((0.0, 0, 0)))
      if roll:
        roll_camera(camera)
      camera.location = perturb(mul(loc, d), shift)

      print_matrix(fo, get_modelview_matrix())
      insert_frame(camera, 2 * i + 1)

    fo.close()

  # Create a bunch of views of the object
  bpy.context.scene.frame_start = 0
  bpy.context.scene.frame_end = 2 * n - 1

  stem = os.path.join(save_dir, '######')
  render(stem)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-m', '--model', dest='model',
                      required=True,
                      help='Path to model obj file.')
  parser.add_argument('-o', '--output_dir', dest='output_dir',
                      required=True,
                      help='Where to output files.')
  parser.add_argument('-s', '--output_size', dest='output_size',
                      required=True,
                      help='Width and height of output in pixels, e.g. 32x32.')
  parser.add_argument('-n', '--num_frames', dest='n', type=int,
                      required=True,
                      help='Number of frames to generate per clip.')

  parser.add_argument('-scale', '--scale', dest='scale', type=float,
                      help='object scaling', default=1)

  parser.add_argument('-perturb', '--perturb', dest='perturb', type=float,
                      help='sphere perturbation', default=0)

  parser.add_argument('-rotate', '--rotate', dest='rotate', action='store_true',
                      help='render rotating test set')

  parser.add_argument('-roll', '--roll', dest='roll', action='store_true',
                      help='add roll')

  parser.add_argument(
      '-fov', '--fov', dest='fov', type=float, required=True,
      help='field of view')

  if '--' not in sys.argv:
    parser.print_help()
    exit(1)

  argv = sys.argv[sys.argv.index('--') + 1:]
  args, _ = parser.parse_known_args(argv)

  random.seed(args.model + str(time.time()) + str(os.getpid()))
  # random.seed(0)

  set_scene(int(args.output_size), args.fov)
  render_obj(
      args.model, args.output_dir, args.n, args.perturb, args.rotate,
      args.roll, args.scale)
  exit()


if __name__ == '__main__':
  main()
