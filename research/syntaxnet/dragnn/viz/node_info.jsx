
/**
 * Template for node info.
 */
import preact from 'preact';
import _ from 'lodash';

const normalCell = {
  'border': 0,
  'border-collapse': 'separate',
  'padding': '2px',
};

/**
 * Style definitions which are directly injected (see README.md comments).
 */
const style = {
  featuresTable: {
    'background-color': 'rgba(255, 255, 255, 0.9)',
    'border': '1px solid #dddddd',
    'border-spacing': '2px',
    'border-collapse': 'separate',
    'font-family': 'roboto, helvectica, arial, sans-serif',
    // Sometimes state strings (`stateHtml`) get long, and because this is an
    // absolutely-positioned box, we need to make them wrap around.
    'max-width': '600px',
    'position': 'absolute',
  },

  heading: {
    'background-color': '#ebf5fb',
    'font-weight': 'bold',
    'text-align': 'center',
    ...normalCell
  },

  normalCell: normalCell,

  featureGroup: (componentColor) => ({
    'background-color': componentColor,
    'font-weight': 'bold',
    ...normalCell
  }),

  normalRow: {
    'border': 0,
    'border-collapse': 'separate',
  },
};

/**
 * Creates table rows that negate IPython/Jupyter notebook styling.
 *
 * @param {?XML|?Array<XML>} children Child nodes. (Recall Preact handles
 *     null/undefined gracefully).
 * @param {!Object} props Any additional properties.
 * @return {!XML} React-y element, representing a table row.
 */
const Row = ({children, ...props}) => (
  <tr style={style.normalRow} {...props}>{children}</tr>);

/**
 * Creates table cells that negate IPython/Jupyter notebook styling.
 *
 * @param {?XML|?Array<XML>} children Child nodes. (Recall Preact handles
 *     null/undefined gracefully).
 * @param {!Object} props Any additional properties.
 * @return {!XML} React-y element, representing a table cell.
 */
const Cell = ({children, ...props}) => (
  <td style={style.normalCell} {...props}>{children}</td>);

/**
 * Construct a table "multi-row" with a shared "header" cell.
 *
 * In ASCII-art,
 *
 * ------------------------------
 *        | row1
 * header | row2
 *        | row3
 * ------------------------------
 *
 * @param {string} headerText Text for the header cell
 * @param {string} headerColor Color of the header cell
 * @param {!Array<XML>} rowsCells Row cells (<td> React-y elements).
 * @return {!Array<XML>} Array of React-y elements.
 */
const featureGroup = (headerText, headerColor, rowsCells) => {
  const headerCell = (
    <td rowspan={rowsCells.length} style={style.featureGroup(headerColor)}>
      {headerText}
    </td>
  );
  return _.map(rowsCells, (cells, i) => {
    return <Row>{i == 0 ? headerCell : null}{cells}</Row>;
  });
};

/**
 * Mini helper to intersperse line breaks with a list of elements.
 *
 * This just replicates previous behavior and looks OK; we could also try spans
 * with `display: 'block'` or such.
 *
 * @param {!Array<XML>} elements React-y elements.
 * @return {!Array<XML>} React-y elements with line breaks.
 */
const intersperseLineBreaks = (elements) => _.tail(_.flatten(_.map(
  elements, (v) => [<br />, v]
)));

export default class NodeInfo extends preact.Component {
  /**
   * Obligatory Preact render() function.
   *
   * It might be worthwhile converting some of the intermediate variables into
   * stateless functional components, like Cell and Row.
   *
   * @param {?Object} selected Cytoscape node selected (null if no selection).
   * @param {?Object} mousePosition Mouse position, if a node is selected.
   * @return {!XML} Preact components to render.
   */
  render({selected, mousePosition}) {
    const visible = selected != null;
    const stateHtml = visible && selected.data('stateInfo');

    // Generates elements for fixed features.
    const fixedFeatures = visible ? selected.data('fixedFeatures') : [];
    const fixedFeatureElements = _.map(fixedFeatures, (feature) => {
      if (feature.value_trace.length == 0) {
        // Preact will just prune this out.
        return null;
      } else {
        const rowsCells = _.map(feature.value_trace, (value) => {
          // Recall `value_name` is a list of strings (representing feature
          // values), but this is OK because strings are valid react elements.
          const valueCells = intersperseLineBreaks(value.value_name);
          return [<Cell>{value.feature_name}</Cell>, <Cell>{valueCells}</Cell>];
        });
        return featureGroup(feature.name, '#cccccc', _.map(rowsCells));
      }
    });

    /**
     * Generates linked feature info from an edge.
     *
     * @param {!Object} edge Cytoscape JS Element representing a linked feature.
     * @return {[XML,XML]} Linked feature information, as table elements.
     */
    const linkedFeatureInfoFromEdge = (edge) => {
      return [
        <Cell>{edge.data('featureName')}</Cell>,
        <Cell>
          value {edge.data('featureValue')} from
          step {edge.source().data('stepIdx')}
        </Cell>
      ];
    };

    const linkedFeatureElements = _.flatten(
      _.map(this.edgeStatesByComponent(), (edges, componentName) => {
        // Because edges are generated by `incomers`, it is guaranteed to be
        // non-empty.
        const color = _.head(edges).source().parent().data('componentColor');
        const rowsCells = _.map(edges, linkedFeatureInfoFromEdge);
        return featureGroup(componentName, color, rowsCells);
      }));

    let positionOrHiddenStyle;
    if (visible) {
      positionOrHiddenStyle = {
        left: mousePosition.x + 20,
        top: mousePosition.y + 10,
      };
    } else {
      positionOrHiddenStyle = {display: 'none'};
    }

    return (
      <table style={_.defaults(positionOrHiddenStyle, style.featuresTable)}>
        <Row>
          <td colspan="3" style={style.heading}>State</td>
        </Row>
        <Row>
          <Cell colspan="3">{stateHtml}</Cell>
        </Row>
        <Row>
          <td colspan="3" style={style.heading}>Features</td>
        </Row>
        {fixedFeatureElements}
        {linkedFeatureElements}
      </table>
    );
  }

  /**
   * Gets a list of incoming edges, grouped by their component name.
   *
   * @return {!Object<string, !Array<!Object>>} Map from component name to list
   *     of edges.
   */
  edgeStatesByComponent() {
    if (this.props.selected == null) {
      return [];
    }
    const incoming = this.props.selected.incomers();  // edges and nodes
    return _.groupBy(incoming.edges(), (edge) => edge.source().parent().id());
  }
}

