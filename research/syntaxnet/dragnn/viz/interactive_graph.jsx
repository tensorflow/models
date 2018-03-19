
/**
 * This is the main view element.
 */
import preact from 'preact';
import NodeInfo from "./node_info.jsx";

/**
 * Style definitions which are directly injected (see README.md comments).
 */
const style = {
  main: {
    'border': '1px solid #cccccc',
    'border-radius': 3,
    'bottom': 3,
    'display': 'flex',
    'flex-direction': 'column',
    'left': 1,
    'position': 'absolute',
    'right': 1,
    'top': 3,
  },
  controlBar: {
    'background-color': '#eeeeee',
    'padding': '6px 3px 6px 3px',
  },
  graph: {
    'flexGrow': 1,
    'min-height': 200,
    'min-width': 200,
  },
  // Filter text input -- negate default bootstrap blocking in IPython
  // notebooks.
  flt: {
    'display': 'inline-block',
    'padding': 3
  },
  // Help/information text in the toolbar -- faded to distinguish from controls.
  helpText: {
    'color': '#666666',
    'font-size': '9pt',
  },
};

export default class InteractiveGraph extends preact.Component {
  constructor() {
    super();
    // Note: This is currently not set, but in the future, it should be
    // user-controllable.
    this.state.horizontal = true;
    this.state.hoverNode = null;
    this.state.mousePosition = {x: 0, y: 0};
  }

  /**
   * Obligatory Preact render function.
   *
   * @param {function} onfilter Function to trigger when the user enters
   *     text. This is called when the user presses <enter>, or de-focuses the
   *     element.
   * @param {?Object} hoverNode Cytoscape node selected (null if no selection).
   * @param {?Object} mousePosition Mouse position, if a node is selected.
   * @return {XML} Preact components to render.
   */
  render({onfilter}, {hoverNode, mousePosition}) {
    return (
      <div style={style.main}>
        <div style={style.controlBar}>
          <label>Fuzzy node filter:&nbsp;</label>
          <input type="text" style={style.flt} onchange={
            (e) => {
              onfilter(e.target.value);
            }
          }/>
          <em style={style.helpText}>
            &nbsp;&bull;&nbsp;
            Hover over a node to view inputs; click to focus/zoom graph.
          </em>
        </div>
        <div ref={(g) => {
          this.graph = g;
        }} style={style.graph}/>
        <NodeInfo selected={hoverNode} mousePosition={mousePosition}/>
      </div>
    );
  }

  /**
   * Calls the `onmount` property, so the caller (visualize.js) can set up the
   * Cytoscape JS graph (since Cytoscape doesn't expose itself as a Preact
   * component).
   */
  componentDidMount() {
    this.props.onmount(this, this.graph);
  }

  /**
   * Public API to update the state, when a node is hovered.
   *
   * @param {!Object} node Cytoscape JS node.
   * @param {!Object} position Position of the mouse.
   */
  showNodeInfo(node, position) {
    this.setState({hoverNode: node, mousePosition: position});
  }

  /**
   * Public API to update the state, when a node is un-hovered.
   */
  hideNodeInfo() {
    this.setState({hoverNode: null});
  }
}

