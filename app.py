import json
from flask import render_template, request, url_for, jsonify, Flask
from predictor import predict_guitar

app = Flask(__name__)

# Instantiating the guitar predictor class. This loads the model
pred = predict_guitar()

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Function for rendering the index page. 
    """
    url = "https://cdn.shopify.com/s/files/1/0022/7193/6612/products/brucespringsteen_vintageblonde2_1024x1024.jpg"

    prediction = pred.predict(url)

    return render_template('index.html',
                           predicted_class=json.dumps(prediction[0]),
                           image=json.dumps(url))

@app.route('/change_image', methods=['GET','POST'])
def change_ad_url():
    """
    Returns prediction for image url
    """
    url = request.args.get('url', '')    

    prediction = pred.predict(url)
   
    return json.dumps([url,prediction[0]])
   

def main():
    app.run(host='0.0.0.0', port=3001, debug=True)

if __name__ == '__main__':
    main()
