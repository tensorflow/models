module.exports = {
  mode : 'production',
  entry : './src/js/main.js',
  output : {
    filename : '../static/js/bundle.js',
  },
  module : {
    rules : [{
            test : /.jsx?$/,
            exclude : /node_modules/,
            use : {
              loader : 'babel-loader',
              options : {
                presets : ['@babel/preset-env'],
                plugins : ['@babel/plugin-transform-runtime']
              }
            }
          }],
  },
};
