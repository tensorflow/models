classdef Layer < handle
  %LAYER Base class for a network layer in a DagNN

  properties (Access = {?dagnn.DagNN, ?dagnn.Layer}, Hidden, Transient)
    net
    layerIndex
  end

  methods
    function outputs = forward(obj, inputs, params)
    %FORWARD Forward step
    %  OUTPUTS = FORWARD(OBJ, INPUTS, PARAMS) takes the layer object OBJ
    %  and cell arrays of inputs and parameters and produces a cell
    %  array of outputs evaluating the layer forward.
      outputs = {} ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutpus)
    %BACKWARD  Bacwkard step
    %  [DERINPUTS, DERPARAMS] = BACKWARD(OBJ, INPUTS, INPUTS, PARAMS,
    %  DEROUTPUTS) takes the layer object OBJ and cell arrays of
    %  inputs, parameters, and output derivatives and produces cell
    %  arrays of input and parameter derivatives evaluating the layer
    %  backward.
      derInputs = {} ;
      derOutputs = {} ;
    end

    function reset(obj)
    %RESET Restore internal state
    %  RESET(OBJ)resets the layer OBJ, clearing any internal state.
    end

    function params = initParams(obj)
    %INIT Initialize layer parameters
    %  PARAMS = INIT(OBJ) takes the layer OBJ and returns a cell
    %  array of layer parameters PARAMS with some initial
    %  (e.g. random) values.
      params = {} ;
    end

    function move(obj, device)
    %MOVE Move data to CPU or GPU
    %  MOVE(DESTINATION) moves the data associated to the layer object OBJ
    %  to either the 'gpu' or the 'cpu'. Note that variables and
    %  parameters are moved automatically by the DagNN object
    %  containing the layer, so this operation affects only data
    %  internal to the layer (e.g. the mask in dropout).
    end

    function forwardAdvanced(obj, layer)
    %FORWARDADVANCED  Advanced driver for forward computation
    %  FORWARDADVANCED(OBJ, LAYER) is the advanced interface to compute
    %  the forward step of the layer.
    %
    %  The advanced interface can be changed in order to extend DagNN
    %  non-trivially, or to optimise certain blocks.

      in = layer.inputIndexes ;
      out = layer.outputIndexes ;
      par = layer.paramIndexes ;
      net = obj.net ;

      inputs = {net.vars(in).value} ;

      % give up if any of the inputs is empty (this allows to run
      % subnetworks by specifying only some of the variables as input --
      % however it is somewhat dangerous as inputs could be legitimaly
      % empty)
      if any(cellfun(@isempty, inputs)), return ; end

      % clear inputs if not needed anymore
      for v = in
        net.numPendingVarRefs(v) = net.numPendingVarRefs(v) - 1 ;
        if net.numPendingVarRefs(v) == 0
          if ~net.vars(v).precious & ~net.computingDerivative & net.conserveMemory
            net.vars(v).value = [] ;
          end
        end
      end

      %[net.vars(out).value] = deal([]) ;

      % call the simplified interface
      outputs = obj.forward(inputs, {net.params(par).value}) ;
      for oi = 1:numel(out)
        net.vars(out(oi)).value = outputs{oi};
      end
    end

    function backwardAdvanced(obj, layer)
    %BACKWARDADVANCED Advanced driver for backward computation
    %  BACKWARDADVANCED(OBJ, LAYER) is the advanced interface to compute
    %  the backward step of the layer.
    %
    %  The advanced interface can be changed in order to extend DagNN
    %  non-trivially, or to optimise certain blocks.
      in = layer.inputIndexes ;
      out = layer.outputIndexes ;
      par = layer.paramIndexes ;
      net = obj.net ;

      inputs = {net.vars(in).value} ;
      derOutputs = {net.vars(out).der} ;
      for i = 1:numel(derOutputs)
        if isempty(derOutputs{i}), return ; end
      end

      if net.conserveMemory
        % clear output variables (value and derivative)
        % unless precious
        for i = out
          if net.vars(i).precious, continue ; end
          net.vars(i).der = [] ;
          net.vars(i).value = [] ;
        end
      end

      % compute derivatives of inputs and paramerters
      [derInputs, derParams] = obj.backward ...
        (inputs, {net.params(par).value}, derOutputs) ;
      if ~iscell(derInputs) || numel(derInputs) ~= numel(in)
        error('Invalid derivatives returned by layer "%s".', layer.name);
      end

      % accumuate derivatives
      for i = 1:numel(in)
        v = in(i) ;
        if net.numPendingVarRefs(v) == 0 || isempty(net.vars(v).der)
          net.vars(v).der = derInputs{i} ;
        elseif ~isempty(derInputs{i})
          net.vars(v).der = net.vars(v).der + derInputs{i} ;
        end
        net.numPendingVarRefs(v) = net.numPendingVarRefs(v) + 1 ;
      end

      for i = 1:numel(par)
        p = par(i) ;
        if (net.numPendingParamRefs(p) == 0 && ~net.accumulateParamDers) ...
              || isempty(net.params(p).der)
          net.params(p).der = derParams{i} ;
        else
          net.params(p).der = vl_taccum(...
            1, net.params(p).der, ...
            1, derParams{i}) ;
          %net.params(p).der = net.params(p).der + derParams{i} ;
        end
        net.numPendingParamRefs(p) = net.numPendingParamRefs(p) + 1 ;
        if net.numPendingParamRefs(p) == net.params(p).fanout
          if ~isempty(net.parameterServer) && ~net.holdOn
            net.parameterServer.pushWithIndex(p, net.params(p).der) ;
            net.params(p).der = [] ;
          end
        end
      end
    end

    function rfs = getReceptiveFields(obj)
    %GETRECEPTIVEFIELDS  Get receptive fields.
    %   RFS = GETRECEPTIVEFIELDS(OBJ) gets the receptive fields
    %   of each output varaibles in each input variable.
    %
    %   A *receptive field* is a structure with fields
    %
    %   - size: size of the receptive field
    %   - stride: stride of the receptive field
    %   - offset: offset of the receptive field
    %
    %   It should be interpreted as follows. Given a pixel of
    %   vertical coordinate u in an output variable OUT(y,...) , the first and last
    %   pixels affecting that pixel in an input variable IN(v,...) are:
    %
    %        v_first = stride(1) * (y - 1) + offset(1) - size(1)/2 + 1
    %        v_last  = stride(1) * (y - 1) + offset(1) + size(1)/2 + 1
    %
    %   RFS is a struct array of such structure, with one row for each
    %   input variable and one column for each output variable, expressing
    %   all possible combinations of inputs and outputs.
      rfs = [] ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      outputSizes = {} ;
    end

    function load(obj, varargin)
    %LOAD Initialize the layer from a paramter structure
    %  LOAD(OBJ, S) initializes the layer object OBJ from the parameter
    %  structure S.  It is the opposite of S = SAVE(OBJ).
    %
    %  LOAD(OBJ, OPT1, VAL1, OPT2, VAL2, ...) uses instead the
    %  option-value pairs to initialize the object properties.
    %
    %  LOAD(OBJ, {OPT1, VAL1, OPT2, VAL2, ...}) is an equivalent form
    %  to the previous call.
      s = dagnn.Layer.argsToStruct(varargin{:}) ;
      for f = fieldnames(s)'
        fc = char(f) ;
        if ~isprop(obj, fc)
          error('No property `%s` for a layer of type `%s`.', ...
            fc, class(obj));
        end;
        obj.(fc) = s.(fc) ;
      end
    end

    function s = save(obj)
    %SAVE Save the layer configuration to a parameter structure
    %  S = SAVE(OBJ) extracts all the properties of the layer object OBJ
    %  as a structure S. It is the oppostie of LOAD(OBJ, S).
    %
    %  By default, properties that are marked as transient,
    %  dependent, abstract, or private in the layer object are not
    %  saved.
      s = struct ;
      m = metaclass(obj) ;
      for p = m.PropertyList'
        if p.Transient || p.Dependent || p.Abstract, continue ; end
        s.(p.Name) = obj.(p.Name) ;
      end
    end

    function attach(obj, net, index)
    %ATTACH  Attach the layer to a DAG.
    %   ATTACH(OBJ, NET, INDEX). Override this function to
    %   configure parameters or variables.
      obj.net = net ;
      obj.layerIndex = index ;
      for input = net.layers(index).inputs
        net.addVar(char(input)) ;
      end
      for output = net.layers(index).outputs
        net.addVar(char(output)) ;
      end
      for param = net.layers(index).params
        net.addParam(char(param)) ;
        p = net.getParamIndex(char(param)) ;
      end
    end

  end % methods

  methods (Static)
    function s = argsToStruct(varargin)
    %ARGSTOSTRUCT  Convert varadic arguments to structure
    %  S = ARGSTOSTRCUT('opt1', val1, ....) converts the list of options
    %  into a structure.
    %
    %  S = ARGSTOSTRUCT({'opt1', val1, ...}) is equivalent to the
    %  previous call.
    %
    %  S = ARGSTOSTRUCT(S) where S is a structure returns S as is.

      if numel(varargin) == 1 && isstruct(varargin{1})
        s = varargin{1} ;
      else
        if numel(varargin) == 1 && iscell(varargin{1})
          args = varargin{1} ;
        else
          args = varargin ;
        end
        s = cell2struct(args(2:2:end),args(1:2:end),2) ;
      end
    end
  end

end % classdef
