"""HuggingFace Space entry point for PolarQuant Gradio demo."""

import os

from polarengine_vllm.cli.cmd_demo import launch_space

launch_space(
    model_id=os.environ.get(
        "MODEL_ID",
        "caiovicentino1/Gemma-4-E4B-it-PolarQuant-Multi/PQ5",
    ),
)
