def generate_gradcam(model, input_tensor):
    with torch.no_grad():
        output, heatmap = model(input_tensor)
    return output, heatmap
