from werkzeug.wrappers import Request, Response
from flask import Flask 
from PIL import Image
from flask import request
from flask import jsonify
from ultralytics import YOLO


# Get request --> send no data 
# Post request --> send data 
# return --> response 
app = Flask(__name__) # app : Flask API 

yolo_model =YOLO('./best.pt')

@app.route("/")
def hello_world():
    return jsonify("Hello from the final session")


@app.route("/predictimage2", methods=["POST"])
def predictimage2():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    try:
        file = request.files['image']
        image = Image.open(file.stream).convert('RGB')
        image = image.resize((800, 800))

        results = yolo_model(image)[0]

        predictions = []
        for box in results.boxes:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            bbox = [int(b) for b in box.xyxy[0].tolist()]

            predictions.append({
                "class": yolo_model.names.get(class_id, str(class_id)),
                "confidence": round(confidence, 2),
                "bbox": bbox
            })

        return jsonify(predictions)

    except Exception as e:
        return jsonify({'error': f'Model inference failed: {str(e)}'}), 500


@app.route("/Media")
def media():
    return "this is media dept."




@app.route("/badrequest")
def sendbadrequest():
    return "Bad Request"


if __name__ == "__main__":
    from werkzeug.serving import run_simple
    run_simple('localhost', 8080, app)
