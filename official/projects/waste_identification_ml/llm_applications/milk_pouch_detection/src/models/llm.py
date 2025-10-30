# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Client for interacting with local LLMs via Ollama."""

import subprocess
import ollama


class LlmModels:
  """Provides an interface to interact with a local LLM via Ollama."""

  def query_image_with_llm(
      self, image_path: str, prompt: str, model_name: str
  ) -> str:
    """Sends an image and a text prompt to a local Ollama LLM.

    Args:
      image_path: Path to the image file.
      prompt: The question or prompt for the LLM.
      model_name: The name of the Ollama model to use (e.g., 'llava').

    Returns:
      The text response from the LLM.
    """
    response: ollama.ChatResponse = ollama.chat(
        model=model_name,
        messages=[{'role': 'user', 'content': prompt, 'images': [image_path]}],
        options={
            'temperature': 0.0,
        },
    )
    return response['message']['content']

  def stop_model(self, model_name: str) -> None:
    """Stops a running Ollama model to free up system resources.

    This function executes the 'ollama stop' command-line instruction.

    Args:
      model_name: The name of the Ollama model to stop.
    """
    print(f'Attempting to stop Ollama model: {model_name}...')
    try:
      result = subprocess.run(
          ['ollama', 'stop', model_name],
          capture_output=True,
          text=True,
          check=False,
      )
      if result.returncode == 0:
        print(f'✅ Successfully sent stop command for model: {model_name}')
      else:
        # This may not be an error if the model wasn't running.
        print(
            'Info: Could not stop model (may not be running):'
            f' {result.stderr.strip()}'
        )
    except FileNotFoundError:
      print(
          "⚠️ 'ollama' command not found. Is Ollama installed and in your PATH?"
      )
    except subprocess.CalledProcessError as e:
      print(f'⚠️ An unexpected error occurred: {e}')
