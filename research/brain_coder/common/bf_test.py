from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""Tests for common.bf."""

import tensorflow as tf

from common import bf  # brain coder


class BfTest(tf.test.TestCase):

  def assertCorrectOutput(self, target_output, eval_result):
    self.assertEqual(target_output, eval_result.output)
    self.assertTrue(eval_result.success)
    self.assertEqual(bf.Status.SUCCESS, eval_result.failure_reason)

  def testBasicOps(self):
    self.assertCorrectOutput(
        [3, 1, 2],
        bf.evaluate('+++.--.+.'))
    self.assertCorrectOutput(
        [1, 1, 2],
        bf.evaluate('+.<.>++.'))
    self.assertCorrectOutput(
        [0],
        bf.evaluate('+,.'))
    self.assertCorrectOutput(
        [ord(char) for char in 'Hello World!\n'],
        bf.evaluate(
            '>++++++++[-<+++++++++>]<.>>+>-[+]++>++>+++[>[->+++<<+++>]<<]>-----'
            '.>->+++..+++.>-.<<+[>[+>+]>>]<--------------.>>.+++.------.-------'
            '-.>+.>+.'))

  def testBase(self):
    self.assertCorrectOutput(
        [1, 4],
        bf.evaluate('+.--.', base=5, input_buffer=[]))

  def testInputBuffer(self):
    self.assertCorrectOutput(
        [2, 3, 4],
        bf.evaluate('>,[>,]<[.<]', input_buffer=[4, 3, 2]))

  def testBadChars(self):
    self.assertCorrectOutput(
        [2, 3, 4],
        bf.evaluate('>,[>,]hello<world[.<]comments',
                    input_buffer=[4, 3, 2]))

  def testUnmatchedBraces(self):
    self.assertCorrectOutput(
        [3, 6, 1],
        bf.evaluate('+++.]]]]>----.[[[[[>+.',
                    input_buffer=[],
                    base=10,
                    require_correct_syntax=False))

    eval_result = bf.evaluate(
        '+++.]]]]>----.[[[[[>+.',
        input_buffer=[],
        base=10,
        require_correct_syntax=True)
    self.assertEqual([], eval_result.output)
    self.assertFalse(eval_result.success)
    self.assertEqual(bf.Status.SYNTAX_ERROR,
                     eval_result.failure_reason)

  def testTimeout(self):
    er = bf.evaluate('+.[].', base=5, input_buffer=[], timeout=0.1)
    self.assertEqual(
        ([1], False, bf.Status.TIMEOUT),
        (er.output, er.success, er.failure_reason))
    self.assertTrue(0.07 < er.time < 0.21)

    er = bf.evaluate('+.[-].', base=5, input_buffer=[], timeout=0.1)
    self.assertEqual(
        ([1, 0], True, bf.Status.SUCCESS),
        (er.output, er.success, er.failure_reason))
    self.assertTrue(er.time < 0.15)

  def testMaxSteps(self):
    er = bf.evaluate('+.[].', base=5, input_buffer=[], timeout=None,
                     max_steps=100)
    self.assertEqual(
        ([1], False, bf.Status.STEP_LIMIT, 100),
        (er.output, er.success, er.failure_reason, er.steps))

    er = bf.evaluate('+.[-].', base=5, input_buffer=[], timeout=None,
                     max_steps=100)
    self.assertEqual(
        ([1, 0], True, bf.Status.SUCCESS),
        (er.output, er.success, er.failure_reason))
    self.assertTrue(er.steps < 100)

  def testOutputMemory(self):
    er = bf.evaluate('+>++>+++>++++.', base=256, input_buffer=[],
                     output_memory=True)
    self.assertEqual(
        ([4], True, bf.Status.SUCCESS),
        (er.output, er.success, er.failure_reason))
    self.assertEqual([1, 2, 3, 4], er.memory)

  def testProgramTrace(self):
    es = bf.ExecutionSnapshot
    er = bf.evaluate(',[.>,].', base=256, input_buffer=[2, 1], debug=True)
    self.assertEqual(
        [es(codeptr=0, codechar=',', memptr=0, memval=0, memory=[0],
            next_input=2, output_buffer=[]),
         es(codeptr=1, codechar='[', memptr=0, memval=2, memory=[2],
            next_input=1, output_buffer=[]),
         es(codeptr=2, codechar='.', memptr=0, memval=2, memory=[2],
            next_input=1, output_buffer=[]),
         es(codeptr=3, codechar='>', memptr=0, memval=2, memory=[2],
            next_input=1, output_buffer=[2]),
         es(codeptr=4, codechar=',', memptr=1, memval=0, memory=[2, 0],
            next_input=1, output_buffer=[2]),
         es(codeptr=5, codechar=']', memptr=1, memval=1, memory=[2, 1],
            next_input=0, output_buffer=[2]),
         es(codeptr=2, codechar='.', memptr=1, memval=1, memory=[2, 1],
            next_input=0, output_buffer=[2]),
         es(codeptr=3, codechar='>', memptr=1, memval=1, memory=[2, 1],
            next_input=0, output_buffer=[2, 1]),
         es(codeptr=4, codechar=',', memptr=2, memval=0, memory=[2, 1, 0],
            next_input=0, output_buffer=[2, 1]),
         es(codeptr=5, codechar=']', memptr=2, memval=0, memory=[2, 1, 0],
            next_input=0, output_buffer=[2, 1]),
         es(codeptr=6, codechar='.', memptr=2, memval=0, memory=[2, 1, 0],
            next_input=0, output_buffer=[2, 1]),
         es(codeptr=7, codechar='', memptr=2, memval=0, memory=[2, 1, 0],
            next_input=0, output_buffer=[2, 1, 0])],
        er.program_trace)


if __name__ == '__main__':
  tf.test.main()
