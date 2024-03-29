from flask import Flask, render_template, Response, request, jsonify
from hashtag_generator import TagGenerator

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')      

@app.route('/generate_tags', methods=['POST'])
def generate_tags():
    text = request.form.get('formData')
    print(text)
    return text
    # keywords = TagGenerator.extract_keywords(text)
    # tags = TagGenerator.generate_tags(keywords)
    # return tags

#render home.html

# @app.route('/cool')
# def index():
#     return render_template('index.html')                                                                   #render index.html

# def gen(camera):
#     while True:
#         frame =VideoCamera.get_frame(camera)  
#                                                                     #call get_frame() function from camera
#         yield (b'--frame\r\n'                                                   #also shows image in bytes format to normal format ;)
#                 b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
#         return frame,camera.video

# @app.route('/video_frame')
# def video_feed():
#     return Response(gen(VideoCamera()),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')       #mimetype is for the browser, we are basically letting browser what type of file it is


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000', debug=True)
    