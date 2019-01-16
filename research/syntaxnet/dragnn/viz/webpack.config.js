
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
    rules: [{
      // Uses some new-style (ECMA 2015) classes ... compile them out.
      test: /\.jsx?$/,

      exclude: /node_modules/,

      use: [{
        loader: 'babel-loader',

        options: {
          presets: [['@babel/preset-env', {targets: 'cover 99.5%'}]],
          plugins: [
            '@babel/plugin-proposal-object-rest-spread',
            ['@babel/plugin-transform-react-jsx', {'pragma': 'preact.h'}],
          ],
        }
      }]
    }]
  }
};

