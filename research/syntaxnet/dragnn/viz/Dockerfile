FROM node:7
WORKDIR /code
COPY package.json /code/
RUN npm install
RUN wget 'https://raw.githubusercontent.com/google/palette.js/master/palette.js'
RUN ln -s src/webpack.config.js webpack.config.js
RUN ln -s src/index.html index.html

# To install packages and then save them to package.json,
#   1. Add something like `RUN npm install webpack-dev-server --save-dev` here.
#   2. After the container is built, run
#
#     docker run --rm -ti dragnn-viz-dev cat package.json | jq . \
#       > $dragnn/viz/package.json

EXPOSE 9000
CMD node_modules/.bin/webpack-dev-server
