
/**
 * @fileoverview This file contains click handlers for a DRAGNN trace graph.
 */
const _ = require('lodash');

/**
 * Handler for when a node is clicked.
 *
 * We first highlight the node's neighbors and 2nd-order neighbors differently,
 * giving them an aesthetically-pleasing fade-out.
 *
 * We then shift all of the components around, such that the larger (2nd-order)
 * selection elements are centered on the screen. This has the effect of making
 * edges between different components more legible, since they are not at as
 * much of an extreme angle.
 *
 * @param {!Object} e Cytoscape event object.
 * @param {!Object} cy Main Cytoscape graph object.
 * @param {boolean} horizontal Whether the layout is horizontal.
 */
const onSelectNode = function(e, cy, horizontal) {
  const node = e.cyTarget;
  // For selecting components, join all children.
  const withChildren = node.union(node.children());
  const selection =
      withChildren.union(withChildren.neighborhood()).filter(':visible');
  const _neighborhood = selection.neighborhood().filter(':visible');
  const nearbyOnly = _neighborhood.difference(selection);
  const nearbyAndSelected = _neighborhood.union(selection);

  // Reset faded, then set it on nodes/edges that should have it.
  cy.batch(() => {
    cy.elements().removeClass('faded-near faded-far');
    nearbyOnly.addClass('faded-near');
    nearbyAndSelected.abscomp().addClass('faded-far');
  });

  // Shift around components so they line up.
  const stepDim = horizontal ? 'x' : 'y';
  if (node.hasClass('step')) {
    const selectedStepNodes = nearbyAndSelected.nodes().filter('.step');
    const nodeGroups =
        _.groupBy(selectedStepNodes, (node) => node.data('parent'));
    const means = _.mapValues(
        nodeGroups,
        (nodes) => _.mean(_.map(nodes, node => node.position()[stepDim])));
    _.each(cy.$('node.component'), (compNode) => {
      if (means[compNode.id()] === undefined) {
        return;
      }
      const offset = _.mean(_.values(means)) - means[compNode.id()];
      _.each(compNode.children(), (node) => {
        node.position(stepDim, node.position()[stepDim] + offset);
      });
    });
  }

  // Zoom to fit the selection
  cy.fit(nearbyAndSelected, 60);
};

class HighlightHandler {
  /**
   * Constructs a new highlight handler.
   *
   * @param {!Object} cy Cytoscape graph controller.
   * @param {!Object} view Preact view component (InteractiveGraph)
   */
  constructor(cy, view) {
    this.cy = cy;
    this.view = view;
    this.mouseoverEdgeIds = [];
    this.debouncedHighlight =
        _.debounce(this.setNewEdgeHighlight.bind(this), 10);
  }

  /**
   * Handles an event.
   *
   * Typically called through debouncedHighlight().
   *
   * @param {?Object} e Cytoscape event from cy.on() handlers; can be null
   *     on mouse-out.
   * @param {?Object} node Cytoscape node for a mouse-over; null to trigger
   *     mouse-out.
   */
  handleEvent(e, node) {
    if (node == null) {
      // mouseout-type handler
      this.view.hideNodeInfo();
      this.debouncedHighlight(this.cy.collection([]));
    } else {
      this.view.showNodeInfo(node, {
        x: e.originalEvent.offsetX,
        y: e.originalEvent.offsetY,
      });
      this.debouncedHighlight(node.neighborhood().edges());
    }
  }

  /**
   * Switches the highlight from one set of edges to the next. Quickly compares
   * the IDs of the new set of edges and the old, so we only update the
   * highlight if it changes.
   *
   * @param {!Object} edges Cytoscape collection of new edges.
   */
  setNewEdgeHighlight(edges) {
    const newEdgeIds = _.map(edges, (e) => e.id());
    newEdgeIds.sort();
    if (!_.isEqual(newEdgeIds, this.mouseoverEdgeIds)) {
      this.cy.edges().removeClass('highlighted-edge');
      edges.addClass('highlighted-edge');
      this.mouseoverEdgeIds = newEdgeIds;
    }
  }
}

/**
 * Adds click/hover handlers to a Cytoscape graph.
 *
 * @param {!Object} cy Main Cytoscape graph object
 * @param {!InteractiveGraph} view Preact view component
 */
export default function setupTraceInteractionHandlers(cy, view) {
  const handler = new HighlightHandler(cy, view);
  const debouncedHandler = handler.handleEvent.bind(handler);

  cy.on('mouseover mousemove', 'node.step', (e) => {
    debouncedHandler(e, e.cyTarget);
  });

  cy.on('mouseout', 'node', () => {
    debouncedHandler(null, null);
  });

  cy.on('tap', 'node', e => {
    // Since onSelectNode is going to move around components, it makes sense
    // to clear the highlight.
    debouncedHandler(null, null);
    onSelectNode(e, cy, view.state.horizontal);
  });

  cy.on('tap', function(e) {
    if (e.cyTarget === cy) {
      cy.elements().removeClass('faded-near faded-far');
    }
  });
};

