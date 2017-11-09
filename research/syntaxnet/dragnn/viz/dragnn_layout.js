
/**
 * @fileoverview Cytoscape layout function for DRAGNN graphs.
 *
 * Currently, the algorithm has 3 stages:
 *
 *   1. Initial layout
 *   2. Spring-based resolution
 *   3. Re-layout based on component order and direction
 *
 * In the future, if we propagated a few more pieces of information, we could
 * probably skip the spring-based step altogether.
 */
'use strict';

const _ = require('lodash');

// default layout options
const defaults = {
  horizontal: true,
  ready() {},  // on layoutready
  stop() {},   // on layoutstop
};

/**
 * Partitions nodes into component nodes and step nodes.
 *
 * @param {!Array} nodes Nodes to partition.
 * @return {!Object<string, Object>} dictionary with two keys, 'component' and
 *     'step', both of which are list of Cytoscape nodes.
 */
function partitionNodes(nodes) {
  // Split nodes into components and steps per component.
  const partition = {component: [], step: []};
  _.each(nodes, function(node) {
    partition[node.hasClass('step') ? 'step' : 'component'].push(node);
  });
  return partition;
}

/**
 * Partitions step nodes by their component name.
 *
 * @param {!Array} nodes Nodes to partition.
 * @return {!Object<string, Object>} dictionary keys as component names,
 *     values as children of that component.
 */
function partitionStepNodes(nodes) {
  const partition = {};
  _.each(nodes, (node) => {
    const key = node.data('parent');
    if (partition[key] === undefined) {
      partition[key] = [node];
    } else {
      partition[key].push(node);
    }
  });
  return partition;
}

/**
 * Initializes the custom Cytoscape layout. This needs to be an old-style class,
 * because of how it's called in Cytoscape.
 *
 * @param {!Object} options Options to initialize with. These will be passed
 *     through to the intermediate "cose" layout.
 */
function DragnnLayout(options) {
  this.options = _.extend({}, defaults, options);
  this.horizontal = this.options.horizontal;
}

/**
 * Calculates the step position, given an effective component index, and step
 * index.
 *
 * @param {number} componentIdx Zero-based (display) index of the component.
 * @param {number} stepIdx Zero-based (display) index of the step.
 * @return {!Object<string, number>} Position dictionary (x and y)
 */
DragnnLayout.prototype.stepPosition = function(componentIdx, stepIdx) {
  return (
      this.horizontal ? {'x': stepIdx * 30, 'y': 220 * componentIdx} :
                        {'x': 320 * componentIdx, 'y': stepIdx * 30});
};

/**
 * The main method for our DRAGNN-specific layout. See module docstring.
 *
 * Cytoscape automatically injects `this.trigger` methods and `options.cy`,
 * `options.eles` variables.
 *
 * @return {DragnnLayout} `this`, for chaining.
 */
DragnnLayout.prototype.run = function() {
  const eles = this.options.eles;  // elements to consider in the layout
  const cy = this.options.cy;

  this.trigger('layoutstart');

  const visible = _.filter(eles.nodes(), function(n) {
    return n.visible();
  });
  const partition = partitionNodes(visible);
  const stepPartition = partitionStepNodes(partition.step);

  // Initialize components as horizontal or vertical "strips".
  _.each(stepPartition, (stepNodes) => {
    _.each(stepNodes, (node, idx) => {
      node.position(this.stepPosition(node.data('componentIdx'), idx));
    });
  });

  // Next do a cose layout, and then run finalLayout().
  cy.layout(_.extend({}, this.options, {
    name: 'cose',
    animate: false,
    ready: this.finalLayout.bind(this, partition, stepPartition, cy)
  }));

  return this;
};

/**
 * Gets a list of components, by their current visual position.
 *
 * @param {!Array} componentNodes Cytoscape component nodes.
 * @return {!Array<string, Object>} List of (componentName, position dict)
 *     pairs.
 */
DragnnLayout.prototype.sortedComponents = function(componentNodes) {
  // Position dictionaries are mutable, so copy them to avoid confusion.
  const copyPosition = (pos) => {
    return {x: pos.x, y: pos.y};
  };

  const componentPositions = _.map(componentNodes, (node) => {
    return [node.id(), copyPosition(node.position())];
  });

  return _.sortBy(componentPositions, (x) => {
    return this.horizontal ? x[1].y : x[1].x;
  });
};

/**
 * Computes the final, fancy layout. This will use two components from the
 * spring model,
 *
 *  - the order of components
 *  - directionality within components
 *
 * and redo layout in a way that's visually appealing (but may not be minimizing
 * distance).
 *
 * @param {!Object<string, Object>} partition Result of partitionNodes().
 * @param {!Object<string, Object>} stepPartition Result of
 *     partitionStepNodes().
 * @param {!Object} cy Cytoscape controller.
 */
DragnnLayout.prototype.finalLayout = function(partition, stepPartition, cy) {
  // Helper to abstract the horizontal vs. vertical layout.
  const compDim = this.horizontal ? 'y' : 'x';
  const stepDim = this.horizontal ? 'x' : 'y';

  const sorted = this.sortedComponents(partition.component);

  // Computes dictionaries from old --> new component positions.
  const newCompPos = _.fromPairs(_.map(sorted, (x, i) => {
    return [x[0], this.stepPosition(i, 0)];
  }));

  // Component --> slope for "step index --> position" function.
  const nodesPerComponent = {};
  const stepSlope = {};

  _.each(stepPartition, (stepNodes) => {
    const nodeOffset = (node) => {
      return node.relativePosition()[stepDim];
    };

    const name = _.head(stepNodes).data('parent');
    const slope =
        (nodeOffset(_.last(stepNodes)) - nodeOffset(_.head(stepNodes))) /
        stepNodes.length;

    nodesPerComponent[name] = stepNodes.length;
    stepSlope[name] =
        Math.sign(slope) * Math.min(300, Math.max(100, Math.abs(slope)));
  });

  // Reset ordering of components based on whether they are actually
  // left-to-right. In the future, we may want to do the whole layout based on
  // the master spec (what remains is slope magnitude and component order); then
  // we can also skip the initial layout and CoSE intermediate layout.
  if (this.options.masterSpec) {
    _.each(this.options.masterSpec.component, (component) => {
      const name = component.name;
      const transitionParams = component.transition_system.parameters || {};
      // null/undefined should default to true.
      const leftToRight = transitionParams.left_to_right != 'false';

      // If the slope isn't going in the direction it should, according to the
      // master spec, reverse it.
      if ((leftToRight ? 1 : -1) != Math.sign(stepSlope[name])) {
        stepSlope[name] = -stepSlope[name];
      }
    });
  }

  // Set new node positions. As before, component nodes auto-size to fit.
  _.each(stepPartition, (stepNodes) => {
    const component = _.head(stepNodes).data('parent');
    const newPos = newCompPos[component];

    _.each(stepNodes, function(node, i) {
      // Keep things near the component centers.
      const x = i - (nodesPerComponent[component] / 2);

      const offset = {};
      offset[compDim] = 40 * Math.log(1.1 + (i % 5)) * (1 - 2 * (i % 2));
      offset[stepDim] = stepSlope[component] * x / 2;

      node.position({'x': newPos.x + offset.x, 'y': newPos.y + offset.y});
    });
  });

  // Set the curvature of edges. For now, we only bend edges within components,
  // by bending them away from the component center.
  _.each(this.options.eles.edges().filter(':visible'), function(edge) {
    const src = edge.source();
    const dst = edge.target();
    const srcPos = src.position();
    const dstPos = dst.position();
    const stepDiff = dstPos[stepDim] - srcPos[stepDim];

    if (src.data('componentIdx') == dst.data('componentIdx')) {
      const avgRelPosition =
          (src.relativePosition()[compDim] + dst.relativePosition()[compDim]);

      // Only bend longer edges.
      if (Math.abs(stepDiff) > 250) {
        const amount = stepDiff / 10;
        const direction = Math.sign(avgRelPosition + 0.001);
        edge.data('curvature', direction * amount);
      }
    }
  });

  // trigger layoutready when each node has had its position set at least once
  this.one('layoutready', this.options.ready);
  this.trigger('layoutready');

  // trigger layoutstop when the layout stops (e.g. finishes)
  this.one('layoutstop', this.options.stop);
  this.trigger('layoutstop');

  // For some reason (not sure yet), this needs to happen on the next tick.
  // (It's not that the component nodes need to resize--that happens even if
  // the selection is limited to node.step).
  setTimeout(() => {
    cy.fit(cy.$('node:visible'), 30);
  }, 10);
};

module.exports = DragnnLayout;

