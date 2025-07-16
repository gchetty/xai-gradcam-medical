
from flask import Flask, request, render_template, send_file
import torch
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt
from models.unetpp import UNetPP
from xai_utils.grad_cam import generate_gradcam
import logging
logging.basicConfig(level=logging.DEBUG)


app = Flask(__name__)
model = UNetPP()
model.eval()


@app.route("/", methods=["GET", "POST"])
def index():
    try:
        if request.method == "POST":
            file = request.files["image"]
            img = Image.open(file).convert("L").resize((512, 512))
            input_tensor = torch.tensor(np.array(img) / 255.0).unsqueeze(0).unsqueeze(0).float()

            # XAI or model inference goes here
            from models.unetpp import UNetPP
            from xai_utils.grad_cam import generate_gradcam

            model = UNetPP()  # or load pretrained model here
            model.eval()

            cam_overlay = generate_gradcam(model, input_tensor)

            # Visualize
            plt.imshow(img, cmap="gray")
            plt.imshow(cam_overlay, cmap="jet", alpha=0.5)
            plt.axis("off")

            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            return send_file(buf, mimetype="image/png")
        return render_template("index.html")
    except Exception as e:
        logging.exception("Error in image processing")
        return f"Error occurred: {e}", 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
