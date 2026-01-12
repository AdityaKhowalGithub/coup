"""
LLM Agent wrapper for HuggingFace models.

Provides a simple interface for generating text using HuggingFace transformers.
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class LLMAgent:
    """Wrapper around HuggingFace transformers for local LLM inference."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-72B-Instruct",
        temperature: float = 0.7,
        device: str = "cuda"  # Use CUDA for GPU
    ):
        """
        Initialize LLM agent.

        Args:
            model_name: HuggingFace model ID (e.g., "Qwen/Qwen2.5-72B-Instruct")
            temperature: Sampling temperature (0.0-1.0)
            device: Device to use ("mps" for M1 Mac, "cuda" for GPU, "cpu" for CPU)
        """
        self.model_name = model_name
        self.temperature = temperature
        self.device = device

        # Get HuggingFace token from environment
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        if not hf_token:
            raise RuntimeError(
                "HUGGINGFACE_TOKEN not found in environment. "
                "Please set it in .env file"
            )

        print(f"Loading model: {model_name}...")
        print(f"Using device: {device}")

        # Load tokenizer and model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                token=hf_token,
                trust_remote_code=True
            )

            # Set padding token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                token=hf_token,
                dtype=torch.float16,  # Use float16 for efficiency
                device_map=device if device != "mps" else None,
                trust_remote_code=True
            )

            # Move to MPS if needed
            if device == "mps":
                self.model = self.model.to("mps")

            self.model.eval()  # Set to evaluation mode

            print(f"Model loaded successfully!")

        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_name}: {e}") from e

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: int = 512
    ) -> str:
        """
        Generate text from prompt.

        Args:
            prompt: Input prompt
            temperature: Override default temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text
        """
        temp = temperature if temperature is not None else self.temperature

        try:
            # Format prompt for chat models
            messages = [{"role": "user", "content": prompt}]

            # Apply chat template if available
            if hasattr(self.tokenizer, "apply_chat_template"):
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                formatted_prompt = prompt

            # Tokenize
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True
            )

            # Move to device
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temp,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            # Decode (skip the input prompt)
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_length:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            return response.strip()

        except Exception as e:
            raise RuntimeError(f"LLM generation failed: {e}") from e

    def __repr__(self) -> str:
        return f"LLMAgent(model={self.model_name}, temp={self.temperature}, device={self.device})"
