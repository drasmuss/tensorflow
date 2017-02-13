# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

"""Gradients for operators defined in data_flow_ops.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import math_ops


@ops.RegisterGradient("DynamicPartition")
def _DynamicPartitionGrads(op, *grads):
  """Gradients for DynamicPartition."""
  data = op.inputs[0]
  indices = op.inputs[1]
  num_partitions = op.get_attr("num_partitions")

  prefix_shape = array_ops.shape(indices)
  original_indices = array_ops.reshape(
      math_ops.range(math_ops.reduce_prod(prefix_shape)), prefix_shape)
  partitioned_indices = data_flow_ops.dynamic_partition(
      original_indices, indices, num_partitions)
  reconstructed = data_flow_ops.dynamic_stitch(partitioned_indices, grads)
  reconstructed = array_ops.reshape(reconstructed, array_ops.shape(data))
  return [reconstructed, None]


@ops.RegisterGradient("DynamicStitch")
def _DynamicStitchGrads(op, grad):
  """Gradients for DynamicStitch."""

  num_values = len(op.inputs) // 2
  indices_grad = [None] * num_values

  def AsInt32(x):
    return (x if op.inputs[0].dtype == dtypes.int32 else
            math_ops.cast(x, dtypes.int32))
  idxs = [AsInt32(array_ops.reshape(op.inputs[i], (-1,)))
          for i in xrange(num_values)]
  if isinstance(grad, ops.IndexedSlices):
    output_shape = array_ops.shape(op.outputs[0])
    output_rows = output_shape[0]
    grad = math_ops.unsorted_segment_sum(grad.values, grad.indices, output_rows)

  values_grad = []
  later_idxs = idxs[-1]
  zeros = array_ops.zeros_like(grad)
  for i in range(num_values - 1, -1, -1):
    if i == num_values - 1:
      v_grad = array_ops.gather(grad, idxs[i])
    else:
      def is_unique(val):
        return math_ops.reduce_all(math_ops.not_equal(val, later_idxs))
      unique = functional_ops.map_fn(is_unique, idxs[i], dtype=dtypes.bool)
      diff_indices = array_ops.where(unique)[:, 0]
      diff_values = array_ops.gather(idxs[i], diff_indices)

      later_idxs = array_ops.concat((diff_values, later_idxs), axis=0)

      n_indices = array_ops.shape(idxs[i])[0]
      v_grad = data_flow_ops.dynamic_stitch(
        [math_ops.range(n_indices), math_ops.cast(diff_indices, dtypes.int32)],
        [zeros[:n_indices], array_ops.gather(grad, diff_values)])

    v_grad = array_ops.reshape(
      v_grad, array_ops.concat((array_ops.shape(op.inputs[i]),
                                array_ops.shape(v_grad)[1:]), 0))

    values_grad = [v_grad] + values_grad

  return indices_grad + values_grad


ops.NotDifferentiable("Queue")
ops.NotDifferentiable("QueueEnqueue")
ops.NotDifferentiable("QueueEnqueueMany")
ops.NotDifferentiable("QueueDequeue")
ops.NotDifferentiable("QueueDequeueMany")
ops.NotDifferentiable("QueueDequeueUpTo")
ops.NotDifferentiable("QueueClose")
ops.NotDifferentiable("QueueSize")

ops.NotDifferentiable("Stack")
ops.NotDifferentiable("StackPush")
ops.NotDifferentiable("StackPop")
ops.NotDifferentiable("StackClose")

ops.NotDifferentiable("GetSessionHandle")
ops.NotDifferentiable("GetSessionTensor")
ops.NotDifferentiable("DeleteSessionTensor")
