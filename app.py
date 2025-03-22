from flask import Flask, render_template, request, send_from_directory
from model.model2 import process_video, process_audio, VideoFeatureExtractor, AudioFeatureExtractor, DeepfakeClassifier
import os
import torch

app = Flask(__name__, static_folder="static", template_folder="templates")

# Directory to temporarily save uploaded videos
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('index.html', content="Upload a video to generate output.")

@app.route('/submit', methods=['POST'])
def handle_submit():
    if 'video' not in request.files:
        return render_template('index.html', content="No video file received. Please upload a valid video.")

    # Get the uploaded video file
    video_file = request.files['video']
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
    video_file.save(video_path)

    try:
        # Process video and audio features using model2.py
        video_frames = process_video(video_path)
        audio_features = process_audio(video_path)

        # Load models from model2.py
        video_model = VideoFeatureExtractor()
        audio_model = AudioFeatureExtractor()
        classifier = DeepfakeClassifier()

        # Set models to evaluation mode
        video_model.eval()
        audio_model.eval()
        classifier.eval()

        # Perform predictions
        with torch.no_grad():
            video_features = video_model(video_frames)
            audio_features = audio_model(audio_features)
            prediction_tensor = classifier(video_features, audio_features)

        # Extract real/fake prediction from the tensor
        real_probability = float(prediction_tensor[0, 0])
        prediction = "Real" if real_probability > 0.50 else "Fake"

        # Clean up temporary files
        os.remove(video_path)

        # Render results in HTML template
        return render_template('index.html', content=f"The uploaded video is predicted as: {prediction}")
    except Exception as e:
        os.remove(video_path)  # Clean up in case of an error
        return render_template('index.html', content=f"Error processing video: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
