# LMI Text Embedding User Guide

Text Embedding refers to the process of converting text data into numerical vectors.
These embeddings capture the semantic meaning of the text and can be used for various
tasks such as semantic search and similarity detection.

The inference process involves:

1. **Loading a Model**: Loading a model from local directory, S3, DJL model zoo or from huggingface repository.
2. **Tokenization**: Breaking down the input text into tokens that the model can understand.
3. **Embeddings**: Passing the tokens through the model to produce embeddings. Embedding is a
multi-dimension vector that could be used for RAG or general embedding search.

LMI supports Text Embedding Inference with the following engines:

- OnnxRuntime
- PyTorch
- Rust
- Python
- vLLM

Currently, the Rust engine provides the best performance for text embedding in LMI.

## Quick Start Configurations

You can leverage LMI Text Embedding inference using the following starter configurations:

### DJL model zoo

You can specify the `djl://` url to load a model from the DJL model zoo.

```
HF_MODEL_ID=djl://ai.djl.huggingface.onnxruntime/BAAI/bge-base-en-v1.5
# Optional
SERVING_BATCH_SIZE=32
```

### environment variables

You can specify the `HF_MODEL_ID` environment variable to load a model from HuggingFace hub, DJL Model Zoo, AWS S3, or a local path. 
DJLServing will download the model from HuggingFace hub and optimize the model with the selected engine at runtime.

```
OPTION_ENGINE=Rust
HF_MODEL_ID=BAAI/bge-base-en-v1.5
# Optional
SERVING_BATCH_SIZE=32
```

You can follow [this example](../deployment_guide/deploying-your-endpoint.md#option-2-configuration---environment-variables)
to deploy a model with environment variable configuration on SageMaker.

### serving.properties

**Rust**

```
engine=Rust
option.model_id=BAAI/bge-base-en-v1.5
translatorFactory=ai.djl.huggingface.translator.TextEmbeddingTranslatorFactory
# Optional
batch_size=32
```

**vLLM**

```
engine=Python
option.model_id=BAAI/bge-base-en-v1.5
option.task=text-embedding
option.entryPoint=djl_python.lmi_vllm.vllm_async_service
option.rolling_batch=disable
option.async_mode=true
# Optional: disable L2 normalization (default: true)
# option.normalize=false
```

You can follow [this example](../deployment_guide/deploying-your-endpoint.md#option-1-configuration---servingproperties)
to deploy a model with serving.properties configuration on SageMaker.

## Deploy model to SageMaker

The following code example demonstrates this configuration UX using the [SageMaker Python SDK](https://github.com/aws/sagemaker-python-sdk).

This example will use the [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) model. 

```python
# Assumes SageMaker Python SDK is installed. For example: "pip install sagemaker"
import sagemaker
from sagemaker.djl_inference import DJLModel

# Setup role and sagemaker session
role = sagemaker.get_execution_role()  # execution role for the endpoint
session = sagemaker.session.Session()  # sagemaker session for interacting with different AWS APIs

# Create the SageMaker Model object.
model_id = "BAAI/bge-base-en-v1.5"

env = {
    "SERVING_MIN_WORKERS": "1", # make sure min and max Workers are equals when deploy model on GPU
    "SERVING_MAX_WORKERS": "1",
}

model = DJLModel(
    model_id=model_id,
    task="text-embedding",
    env=env,
)

# Deploy your model to a SageMaker Endpoint and create a Predictor to make inference requests
instance_type = "ml.g4dn.2xlarge"
endpoint_name = sagemaker.utils.name_from_base("lmi-text-embedding")

predictor = model.deploy(initial_instance_count=1,
             instance_type=instance_type,
             endpoint_name=endpoint_name,
             )

# Make an inference request against the endpoint
predictor.predict(
    {"inputs": "What is Deep Learning?"}
)
```

The full notebook is available [here](https://github.com/deepjavalibrary/djl-demo/blob/master/aws/sagemaker/large-model-inference/sample-llm/text_embedding_deploy_bert.ipynb).

## Available Environment Variable Configurations

The following environment variables are exposed as part of the UX:

**HF_MODEL_ID**

This configuration is used to specify the location of your model artifacts.

**HF_REVISION**

If you are using a model from the HuggingFace Hub, this specifies the commit or branch to use when downloading the model.

This is an optional config, and does not have a default value. 

**HF_MODEL_TRUST_REMOTE_CODE**

If the model artifacts contain custom modeling code, you should set this to true after validating the custom code is not malicious.
If you are using a HuggingFace Hub model id, you should also specify `HF_REVISION` to ensure you are using artifacts and code that you have validated.

This is an optional config, and defaults to `False`.

**OPTION_ENGINE**

This option represents the Engine to use, values include `OnnxRuntime`, `PyTorch`, `Rust`, etc.

**SERVING_BATCH_SIZE**

This option represents the dynamic batch size.

This is an optional config, and defaults to `1`.

**SERVING_MIN_WORKERS**

This option represents minimum number of workers.

This is an optional config, and defaults to `1`.

**SERVING_MAX_WORKERS**

This option represents the maximum number of workers.

This is an optional config, and default is `#CPU` for CPU, GPU default is `2`.

When running Text Embedding task on GPU, benchmarking result shows `SERVING_MAX_WORKERS=1` gives better performance.
We recommend to use same value for `SERVING_MIN_WORKERS` and `SERVING_MAX_WORKERS` on GPU to avoid worker scaling overhead.

### Additional Configurations

Additional configurations are available to further tune and customize your deployment.
These configurations are covered as part of the advanced deployment guide [here](../deployment_guide/configurations.md).

## API Schema

### Request Schema

Request Body Fields:

| Field Name   | Field Type                                    | Required | Possible Values                                                                                                                                     |
|--------------|-----------------------------------------------|----------|-----------------------------------------------------------------------------------------------------------------------------------------------------|
| `inputs`     | string, array of strings                      | yes      | example: "What is Deep Learning", ["What is Deep Learning", "How many ways can I peel an orange"]                                                   |

Example request using curl:

```
curl -X POST http://127.0.0.1:8080/invocations \
  -H 'Content-Type: application/json' \
  -d '{"inputs":"What is Deep Learning?"}'
```

### Response Schema

The response is returned as an array.

Example response:

```
[
  0.0059961616,
  -1.0498056,
  0.040412642,
  -0.2044975,
  0.8382087,
  ...
]
```

#### Error Responses

When using dynamic batching, errors are returned with HTTP response code 400 and content:

``` 
{
  "code": 400,
  "type": "TranslateException",
  "message": "Missing \"inputs\" in json."
}
```

## gRPC Embed

`Inference.Embed` is a unary RPC on the service that already serves `Ping` and `Predict`. Model configuration, HTTP `POST /predictions/{model}` and `POST /invocations`, and gRPC `Predict` stay JSON. The Embed call does not use JSON on the way in or the way out.

`EmbedRequest.inputs` is one or more strings. The service writes that `EmbedRequest` to the worker with `toByteArray`, content type `application/x-protobuf`. The worker parses the same generated message. An empty `inputs` list, or a blank string, is `INVALID_ARGUMENT` and is not sent to the model. A missing model is `NOT_FOUND`.

The worker reply Embed reads is a generated `EmbedResponse`, parsed with `parseFrom`. `embeddings` carries one `Embedding` per row, with `index` and a packed `float` `vector`. Those packed bytes are the engine tensor's float32 buffer, copied into the field. A missing or invalid `EmbedResponse` is `FAILED_PRECONDITION`. A worker that returns only the JSON matrix does not succeed as Embed. HTTP and `Predict` still return that JSON matrix. The vLLM HTTP formatter keeps writing it for those clients. The Embed formatter does not.

```java
EmbedResponse response = client.embed(modelName, List.of("What is Deep Learning?"));
Embedding row = response.getEmbeddings(0);
List<Float> vector = row.getVectorList();
```

`Embed` runs on the workers already configured for that model. On GPU, keep the single-worker setting from the deployment section above. Rust, ONNX, and PyTorch models keep returning JSON for HTTP and `Predict`. They succeed on Embed only when they write an `EmbedResponse`.
