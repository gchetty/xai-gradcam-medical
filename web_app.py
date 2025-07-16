
from flask import Flask, request, render_template, send_file
import torch
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt
from models.unetpp import UNetPP
from xai_utils.grad_cam import generate_gradcam

app = Flask(__name__)
model = UNetPP()
model.eval()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]
        img = Image.open(file).convert("L").resize((256, 256))
        input_tensor = torch.tensor(np.array(img) / 255.0).unsqueeze(0).unsqueeze(0).float()
        output, heatmap = generate_gradcam(model, input_tensor)
        plt.imshow(img, cmap="gray")
        plt.imshow(heatmap, cmap="jet", alpha=0.5)
        plt.axis("off")
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        return send_file(buf, mimetype="image/png")
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
