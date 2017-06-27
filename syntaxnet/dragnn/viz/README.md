# DRAGNN trace visualizer

This mini project builds an interactive JavaScript-based visualization for
DRAGNN traces.

## Using the visualizer

We'll check in a pre-compiled version of the visualization, once it isn't
changing much. Then you can just inject it in an IPython notebook; given a JSON
trace string `master_trace_json`, and the pre-compiled Javascript in
`library_js`, the following Python code will display the visualization,

```python
from IPython.display import HTML

HTML("""
<div id="{elt_id}" style="width: 100%; min-width: 200px; height: 700px;">
</div>
<script type='text/javascript'>
{library_js}

visualizeToDiv({json}, "{elt_id}");
</script>
""".format(library_js=library_js, json=master_trace_json, elt_id="foo-123"))
```

## Development

### Running the development server

To run the development server, first install Docker, then run the `develop.sh`
script in this directory. Then, open http://127.0.0.1:9000/; you should see a
sample graph. Whenever you edit one of the `.js` or `.jsx` files, the browser
should auto-refresh. If your code has an error, it should display in the
development console (press ctrl+shift+j to open it in Chrome).

If you need to add a new package, it's recommended to add a line like

```
RUN npm-install my-new-package --save-dev
```

to the Dockerfile. If you immediately modify `package.json`, by contrast, Node
will go re-download everything (this is more a function of Docker caching than
Node). Please make sure that the new package's license is compatible with
DRAGNN's license.

Modifying packages or the Webpack configuration will require a restart of the
development server. Simply send SIGINT (ctrl+c in a shell), and re-run
`develop.sh`.

### General library use and code structure

Most of the visualization is currently based on cytoscape.js. On the downside,
Cytoscape does not render edges as beautifully as GraphViz (the latter is great
at minimizing edge crossings). However, its interactivity may be an advantage
for visualizing larger graphs.

Element layout uses Preact, a React-style templating language. It claims to be
small, modern, and high-performing. To style elements, we directly attach their
styles using Preact code. This avoids CSS namespace pollution, which is very
important because our graphs are expected to cohabit with other elements in an
IPython notebook or such.

#### Main invocation

The base script is `visualize.js`, and its main function is `visualizeToDiv`
(client usage is demonstrated in the "Using the visualizer" section).

#### Layout

Most complexity is currently in the layout. Currently, we attempt to communicate
ordering of step nodes within each component, and also arrange the components in
a way that looks sensible. Currently, this is done by running an initial layout,
then running Cytoscape's spring solver ("cose") to determine the best ordering
of components and directionality of steps within each component, and finally
running the fancier layout engine.

In the future, it would probably be nice to remove the spring solver (and
initial layout). We should pass through the master specification, so the layout
has information about which components are left-to-right vs. right-to-left. We
can determine the order of components by looking at the number of edges from one
component to the other.
