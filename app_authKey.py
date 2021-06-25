from flask import Flask, request, render_template, jsonify
from io import BytesIO
from captchaModel.model import load_model, inference
# from waitress import serve
import numpy as np
import base64
from PIL import Image
from waitress import serve
from paste.translogger import TransLogger
import time
import math

sess, dectection_graph = load_model()
app = Flask(__name__)

# app.config['SECRET_KEY'] = 'secret key here'
# app.config['HOST'] = '0.0.0.0'
# app.config['DEBUG'] = True
# app.config["CACHE_TYPE"] = "null"

# change to "redis" and restart to cache again

# some time later
# cache.init_app(app)


# @app.route('/')
# def home():
#     return "Hello World"

# def shutdown_server():
#     func = request.environ.get('werkzeug.server.shutdown')
#     if func is None:
#         raise RuntimeError('Not running with the Werkzeug Server')
#     func()

# @app.route('/shutdown', methods=['POST'])
# def shutdown():
#     shutdown_server()
#     return 'Server shutting down...'
    

@app.route('/api/predict/', methods=['GET', 'POST','DELETE', 'PATCH'])
def api_predict():   
    # print(request.is_json) 
    # image_base64 = request.form['image']
    # image = base64.b64decode(image_base64)
    # image = Image.open(BytesIO(image))
    # if image.mode != "RGB":
    #     image.convert("RGB")

    # image_arr = np.array(image, dtype=np.uint8)
    # # print(image_arr)
    # start_date = time.time()
    # res = inference(sess, dectection_graph, image_arr)
    # end_date = time.time()
    # result_date = start_date - end_date
    # # print(res)
    # return jsonify(answer=res, status=True, captcha=res, time=format(math.floor(result_date)).replace("-","") + "s")
    # return {
    #             "captcha": res, 
    #             "time": format(math.floor(result_date)).replace("-","") + "s"}

    # return jsonify(answer=res, status=True)
    apiKeys = []
    with open('key_api.txt', 'r') as listApiKeys:
        apiKeys = [listApiKey.rstrip() for listApiKey in listApiKeys.readlines()]
    # print(apiKey)
    # apiKey = 'Pong1129'

        
    if request.is_json:
        data_json = request.get_json()
        if data_json['api_key'] in apiKeys:
            print('found')
            try:
                if data_json:
                    image_base64 = data_json['image']
                    image = base64.b64decode(image_base64)
                    image = Image.open(BytesIO(image))
                    if image.mode != "RGB":
                        image.convert("RGB")
                    
                    image_arr = np.array(image, dtype=np.uint8)
                    # print(image_arr)
                    res = inference(sess, dectection_graph, image_arr)

                    return jsonify(answer=res, status=True)
            except Exception as e:
                return jsonify(answer="", status=False)
        else:
            return jsonify(answer="NoApiKey", status=False, text='Please Enter API KEY') 
    else:
        image_base64 = request.form['image']
        image = base64.b64decode(image_base64)
        image = Image.open(BytesIO(image))
        if image.mode != "RGB":
            image.convert("RGB")

        image_arr = np.array(image, dtype=np.uint8)
        # print(image_arr)
        start_date = time.time()
        res = inference(sess, dectection_graph, image_arr)
        end_date = time.time()
        result_date = start_date - end_date
        # print(res)
        return {
                    "captcha": res, 
                    "time": format(math.floor(result_date)).replace("-","") + "s"}


            # base64 = request.form['base64']
            # start_date = time.time()
            # captcha = Captcha_detection(base64)
            # end_date = time.time()
            # result_date = start_date - end_date
            # return {
            #     "captcha": captcha, 
            #     "time": format(math.floor(result_date)).replace("-","") + "s"


@app.route('/' , methods=['GET', 'POST','DELETE', 'PATCH'])
def home():
    base64few = str(request.form['base64'])
    image = base64.b64decode(base64few)
    image = Image.open(BytesIO(image))
    if image.mode != "RGB":
        image.convert("RGB")
    image_arr = np.array(image, dtype=np.uint8)
    start_date = time.time()
    #captcha = Captcha_detection(base64)
    res = inference(sess, dectection_graph, image_arr)
    end_date = time.time()
    result_date = start_date - end_date
    
    #print('I have money {:,.2f} baht'.format(result_date))
    
    return {
        "captcha": res, 
        "time": '{:,.2f}s'.format(result_date).replace("-","")
    }


@app.route('/api/images/', methods=['POST'])
def api_images():
    if request.is_json:
        try:
            data_json = request.get_json()
            if data_json:
                image_base64 = data_json['image']
                image = base64.b64decode(image_base64)
                image = Image.open(BytesIO(image))
                if image.mode != "RGB":
                    image.convert("RGB")
                
                image_arr = np.array(image, dtype=np.uint8)
                # print(image_arr)
                res = inference(sess, dectection_graph, image_arr)

                return jsonify(answer=res, status=True)
        except Exception as e:
            return jsonify(answer="", status=False)
    return jsonify(answer="", status=False) 

@app.route('/api/imgArr/', methods=['POST'])
def api_imgArr():
    print(request.get_json())


if __name__ == "__main__":
    # app.run(debug=True)
    app.debug = False
    # app.config["CACHE_TYPE"] = "null"
    app.run(host = '0.0.0.0',port=8080)
    # serve(TransLogger(app, setup_console_handler=False), threads=20, host = '0.0.0.0',port=5000)
    # serve(TransLogger(app, setup_console_handler=False), threads=40, host = '0.0.0.0',port=5000)

