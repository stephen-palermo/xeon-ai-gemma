# xeon-ai-gemma

## Quick start

1. Get access to the [OpenVINO Gemma model](https://huggingface.co/OpenVINO/gemma-4-E4B-it-int8-ov) on Hugging Face and create a user access token.
2. From this directory, run the following commands. Replace `your_token_here` with your Hugging Face token:

```bash
export HF_TOKEN="your_token_here"
bash setup.sh
source openvino_env/bin/activate
python3 ./run_gemma.py
```

The default command expects an image named `image.png` in this directory. To use another image or prompt:

```bash
python3 ./run_gemma.py --image photo.jpg --prompt "Describe this image."
```

The model is downloaded from Hugging Face on the first run and cached for later runs.