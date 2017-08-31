%VL_TMOVE  Utility to move tensors across GPUs
%   VL_TMOVE() is the point of entry to MatConvNet's subsystem for
%   moving efficiently tensors across GPUs and MATLAB
%   processes. Currently it is only supported in Linux and macOS.
%
%   The system works by allocating a number of tensors, as specified
%   by the argument FORMAT given below, with two functions, `push` and
%   `pull`, to, respectively, set the value of one of the tensors and
%   retrieving it.
%
%   The system allows a number of MATLAB instances to exchange tensor
%   data in a coordinated manner. This means that, at each given
%   cycle, each MATLAB instance pushes a new value for a tensor, the
%   values are accumulated by the system in parallel using a separated
%   thread, and, in a second time, each MATLAB instance retrieves the
%   updated value. Importantly, `push` operations are non-blocking, so
%   that MATLAB can proceed with other computations as tensors are
%   exchanged.
%
%   Usually, VL_TMOVE() is used in combination with a MATLAB
%   parallel pool. In this case, each MATLAB process is known as a
%   "lab" and receives an index `labindex`, from 1 to the number of
%   labs. In a pool there are `numlabs` MATLAB instances in total, as
%   specified upon pool creation. The typical setup is to assign a
%   different MATLAB instance to each of a group of GPUs.
%
%   VL_TMOVE() uses indexes to identify different MATLAB processes
%   in the pool. While these are effectively independent of the MATLAB
%   pool lab indexes, it is convenient to use the same codes for both
%   systems.
%
%   The system is initialized by specifying a FORMAT (table of
%   tensors), a lab code LABINDEX, and the total number of labs in the
%   pool NUMLABS. Processes are assumed to run on the same local host
%   (this restriction may be relaxed in the future).
%
%   FORMAT has the same structure as the MATLAB's own function
%   `mempmapfile()`. For example, the following FORMAT declares two
%   tensors, called `x0`, and `x1`, of size 1x1 (resp. 10x5), `single`
%   (`double`) storage class.
%
%       format = {'single', [1  1], 'x0' ;
%                 'double', [10 5], 'x1' }
%
%   As ane extension, it is possible to declare all or some of the
%   tensors as GPU ones, by adding a fourth column to FORMAT:
%
%       format = {'single', [1  1], 'x0', 'cpu' ;
%                 'double', [10 5], 'x1', 'gpu' }
%
%   Push and pull operations are required to use arrays that match the
%   specifications exactly, including being a CPU or GPU array
%   (i.e. VL_TMOVE() never attempts any implicit conversion).
%
%   VL_TMOVE(COMMAND,...) accepts the following commands:
%
%   - `vl_tmove('init',format,labindex,numlabs)`. This call prepares
%     the system for exchanging data for the specified tensor list
%     `format`, the given lab `labindex` and the total number of labs
%     `numlabs`.
%
%   - `vl_tmove('push', name, value)` pushes the new `value` of the
%     tensor `name`.
%
%   - `x = vl_tmove('pull', name)` does the opposite and retrieves the
%     (updated) value of the tensor `name`.
%
%   - `vl_tmove('reset')` resets the system, including closing down
%     any existing connection between MATLAB instances and freeing all
%     memory.
%
%   Commands may take the following options by appending them to the
%   list of parameters:
%
%   Verbose:: not specified
%     If specified, it increases by one the verbosity level (the
%     option can be repeated).
%
%   InPlace::
%     If specified, updates any GPU array in place (CPU arrays are
%     always processed by copying). It is a (slightly) unsafe
%     operation. It must be used with both `push` and `pull`
%     commands. A pushed GPU array must not be deleted before is
%     pulled again. If the array is deleted between a `push` and a
%     `pull`, the system may write unallocated GPU memory.

% Copyright (C) 2016 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).
