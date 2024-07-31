'use strict';

const gulp = require('gulp');
const $ = require('gulp-load-plugins')();

require('es6-promise').polyfill();

const webpack = require('webpack');
const webpackStream = require('webpack-stream');
const webpackConfig = require('./webpack.config');

const src_paths = {
  sass: ['src/scss/*.scss'],
  script: ['src/js/*.js'],
};

const dest_paths = {
  style: 'static/css/',
  script: 'static/js/',
};

/**
 * Lint Sass files.
 * @returns {!Stream} Gulp stream.
 */
function lint_sass() {
  return gulp.src(src_paths.sass)
    .pipe($.plumber({
      errorHandler: function(err) {
        console.log(err.messageFormatted);
        this.emit('end');
      }
    }))
    .pipe($.stylelint({
      config: {
        extends: [
          'stylelint-config-recommended',
          'stylelint-scss',
          'stylelint-config-recommended-scss'
        ],
        rules: {
          'block-no-empty': null,
          'no-descending-specificity': null
        }
      },
      reporters: [{
        formatter: 'string',
        console: true
      }]
    }));
}

/**
 * Compile Sass files to CSS, autoprefix, and minify.
 * @returns {!Stream} Gulp stream.
 */
function style_sass() {
  return gulp.src(src_paths.sass)
      .pipe($.plumber({
        errorHandler: function(err) {
          console.log(err.messageFormatted);
          this.emit('end');
        }
      }))
      .pipe($.sass({outputStyle: 'expanded'}).on('error', $.sass.logError))
      .pipe($.autoprefixer({cascade: false}))
      .pipe(gulp.dest(dest_paths.style))
      .pipe($.cssnano())
      .pipe($.rename({suffix: '.min'}))
      .pipe(gulp.dest(dest_paths.style));
}

/**
 * Lint JavaScript files with ESLint.
 * @returns {!Stream} Gulp stream.
 */
function lint_eslint() {
  return gulp.src(src_paths.script)
      .pipe($.eslint.format())
      .pipe($.eslint.failAfterError());
}

/**
 * Bundle JavaScript files using Webpack.
 * @returns {!Stream} Gulp stream.
 */
function script() {
  return webpackStream(webpackConfig, webpack)
      .on('error',
          function(e) {
            this.emit('end');
          })
      .pipe(gulp.dest('dist'));
}

/**
 * Watch files for changes and run appropriate tasks.
 * @param {function()} done - Callback function.
 */
function watch_files(done) {
  gulp.watch(src_paths.sass).on('change', gulp.series(lint_sass, style_sass));
  gulp.watch(src_paths.script).on('change', gulp.series(lint_eslint, script));
}

exports.lint = gulp.parallel(lint_sass, lint_eslint);
exports.style = style_sass;
exports.script = script;
exports.watch = watch_files;
exports.default = gulp.series(
    gulp.parallel(lint_sass, lint_eslint), gulp.parallel(style_sass, script));
