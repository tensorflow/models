# Copyright 2018 The TensorFlow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for config_util.configdict."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest

from astronet.util import configdict


class ConfigDictTest(absltest.TestCase):

  def setUp(self):
    super(ConfigDictTest, self).setUp()
    self._config = configdict.ConfigDict({
        "int": 1,
        "float": 2.0,
        "bool": True,
        "str": "hello",
        "nested": {
            "int": 3,
        },
        "double_nested": {
            "a": {
                "int": 3,
            },
            "b": {
                "float": 4.0,
            }
        }
    })

  def testAccess(self):
    # Simple types.
    self.assertEqual(1, self._config.int)
    self.assertEqual(1, self._config["int"])

    self.assertEqual(2.0, self._config.float)
    self.assertEqual(2.0, self._config["float"])

    self.assertTrue(self._config.bool)
    self.assertTrue(self._config["bool"])

    self.assertEqual("hello", self._config.str)
    self.assertEqual("hello", self._config["str"])

    # Single nested config.
    self.assertEqual(3, self._config.nested.int)
    self.assertEqual(3, self._config["nested"].int)
    self.assertEqual(3, self._config.nested["int"])
    self.assertEqual(3, self._config["nested"]["int"])

    # Double nested config.
    self.assertEqual(3, self._config["double_nested"].a.int)
    self.assertEqual(3, self._config["double_nested"]["a"].int)
    self.assertEqual(3, self._config["double_nested"].a["int"])
    self.assertEqual(3, self._config["double_nested"]["a"]["int"])

    self.assertEqual(4.0, self._config.double_nested.b.float)
    self.assertEqual(4.0, self._config.double_nested["b"].float)
    self.assertEqual(4.0, self._config.double_nested.b["float"])
    self.assertEqual(4.0, self._config.double_nested["b"]["float"])

    # Nonexistent parameters.
    with self.assertRaises(AttributeError):
      _ = self._config.nonexistent

    with self.assertRaises(KeyError):
      _ = self._config["nonexistent"]

  def testSetAttribut(self):
    # Overwrite existing simple type.
    self._config.int = 40
    self.assertEqual(40, self._config.int)

    # Overwrite existing nested simple type.
    self._config.nested.int = 40
    self.assertEqual(40, self._config.nested.int)

    # Overwrite existing nested config.
    self._config.double_nested.a = {"float": 50.0}
    self.assertIsInstance(self._config.double_nested.a, configdict.ConfigDict)
    self.assertEqual(50.0, self._config.double_nested.a.float)
    self.assertNotIn("int", self._config.double_nested.a)

    # Set new simple type.
    self._config.int_2 = 10
    self.assertEqual(10, self._config.int_2)

    # Set new nested simple type.
    self._config.nested.int_2 = 20
    self.assertEqual(20, self._config.nested.int_2)

    # Set new nested config.
    self._config.double_nested.c = {"int": 30}
    self.assertIsInstance(self._config.double_nested.c, configdict.ConfigDict)
    self.assertEqual(30, self._config.double_nested.c.int)

  def testSetItem(self):
    # Overwrite existing simple type.
    self._config["int"] = 40
    self.assertEqual(40, self._config.int)

    # Overwrite existing nested simple type.
    self._config["nested"].int = 40
    self.assertEqual(40, self._config.nested.int)

    self._config.nested["int"] = 50
    self.assertEqual(50, self._config.nested.int)

    # Overwrite existing nested config.
    self._config.double_nested["a"] = {"float": 50.0}
    self.assertIsInstance(self._config.double_nested.a, configdict.ConfigDict)
    self.assertEqual(50.0, self._config.double_nested.a.float)
    self.assertNotIn("int", self._config.double_nested.a)

    # Set new simple type.
    self._config["int_2"] = 10
    self.assertEqual(10, self._config.int_2)

    # Set new nested simple type.
    self._config.nested["int_2"] = 20
    self.assertEqual(20, self._config.nested.int_2)

    self._config.nested["int_3"] = 30
    self.assertEqual(30, self._config.nested.int_3)

    # Set new nested config.
    self._config.double_nested["c"] = {"int": 30}
    self.assertIsInstance(self._config.double_nested.c, configdict.ConfigDict)
    self.assertEqual(30, self._config.double_nested.c.int)

  def testDelete(self):
    # Simple types.
    self.assertEqual(1, self._config.int)
    del self._config.int

    with self.assertRaises(AttributeError):
      _ = self._config.int

    with self.assertRaises(KeyError):
      _ = self._config["int"]

    self.assertEqual(2.0, self._config["float"])
    del self._config["float"]

    with self.assertRaises(AttributeError):
      _ = self._config.float

    with self.assertRaises(KeyError):
      _ = self._config["float"]

    # Nested config.
    self.assertEqual(3, self._config.nested.int)
    del self._config.nested

    with self.assertRaises(AttributeError):
      _ = self._config.nested

    with self.assertRaises(KeyError):
      _ = self._config["nested"]


if __name__ == "__main__":
  absltest.main()
