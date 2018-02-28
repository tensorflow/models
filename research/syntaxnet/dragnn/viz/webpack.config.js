
const dist_path = '/code/src';

module.exports = {
  context: '/code',
  entry: './src/visualize.js',
  output: {path: dist_path, filename: 'bundle.js'},
  devServer: {
    contentBase: dist_path,
    // We use Docker for host restriction (see develop.sh's -p argument to
    // the `docker run` invocation). Due to how Docker munges host names, this
    // can't be restricted to localhost.
    host: '0.0.0.0',
    port: 9000,
  },
  module: {
    loaders: [{
      // Uses some new-style (ECMA 2015) classes ... compile them out.
      test: /\.jsx?$/,
      exclude: /node_modules/,
      loader: 'babel-loader',
      query: {
        presets: ['es2015'],
        plugins: [
          'transform-object-rest-spread',
          ['transform-react-jsx', {'pragma': 'preact.h'}],
        ],
      }
    }]
  }
};

