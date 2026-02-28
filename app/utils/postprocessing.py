import numpy as np

CLASS_MAP = {
    0: "Glioma",
    1: "Meningioma",
    2: "Pituitary",
    3: "No Tumor"
}

def postprocess(output):
    probabilities = np.exp(output) / np.sum(np.exp(output))
    top_class = int(np.argmax(probabilities))
    confidence = float(probabilities[0][top_class])
    
    label = CLASS_MAP.get(top_class, "Unknown")
    return label, confidence
