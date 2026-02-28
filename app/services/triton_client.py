import tritonclient.http as httpclient
from app.core.config import TRITON_URL, MODEL_NAME

client = httpclient.InferenceServerClient(url=TRITON_URL)


def infer(image_array):
    inputs = httpclient.InferInput("input", image_array.shape, "FP32")
    inputs.set_data_from_numpy(image_array)

    outputs = httpclient.InferRequestedOutput("output")

    response = client.infer(
        model_name=MODEL_NAME,
        inputs=[inputs],
        outputs=[outputs],
    )

    return response.as_numpy("output")
