#!/usr/bin/env python3
"""Publish natural-language scene descriptions from a ROS 2 camera stream."""

# Section: Keep image conversion and model execution compatible with the existing Qwen3-VL camera tools.
from __future__ import annotations

# Section: Store the latest frame safely while inference runs outside the ROS callback thread.
from threading import Lock, Thread

# Section: Validate timer values without constructing temporary tensors.
import math

# Section: Preserve generic frame typing without requiring NumPy at import time.
from typing import Any

# Section: Convert ROS image messages into RGB arrays for the vision-language model.
from cv_bridge import CvBridge

# Section: Use PIL images because the Qwen3-VL processor accepts them directly in multimodal messages.
from PIL import Image

# Section: Provide the ROS 2 node, timer, and lifecycle APIs.
import rclpy
from rclpy.node import Node

# Section: Receive camera frames and publish the generated scene description as plain text.
from sensor_msgs.msg import Image as RosImage
from std_msgs.msg import String

# Section: Load the local Qwen3-VL-compatible model and processor used by this workspace.
import torch
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration


# Section: Use the camera topic requested by the user as the default image input.
DEFAULT_CAMERA_TOPIC = "/camera/image_raw"

# Section: Publish the latest natural-language description on a dedicated ROS topic.
DEFAULT_DESCRIPTION_TOPIC = "/scene_description"

# Section: Use the local model already present in this workspace unless overridden by a ROS parameter.
DEFAULT_MODEL_PATH = "/home/yanjiun68/ai/models/UnifoLM-ER-Flow"

# Section: Accept only ordinary colour image encodings for scene-description inference.
SUPPORTED_COLOR_ENCODINGS = frozenset(("rgb8", "bgr8", "rgba8", "bgra8"))

# Section: Ask the model for a concise, repeatable description suitable for downstream speech or logging.
SCENE_DESCRIPTION_PROMPT = (
    "Describe this camera scene in one concise paragraph. "
    "Mention visible people, important objects, their approximate positions, and notable activity. "
    "Only describe what is clearly visible; do not guess names, identities, text, or hidden objects. "
    "If the scene is unclear, say so. Return plain text only."
)


class SceneDescriptionNode(Node):
    """ROS 2 node that periodically describes the newest frame from /camera/image_raw."""

    def __init__(self) -> None:
        """Special function: Configure ROS interfaces, load the model, and start periodic inference."""
        # Section: Give the node a stable name for ROS graph inspection.
        super().__init__("scene_description_node")

        # Section: Declare configurable interfaces and inference settings for launch-file overrides.
        self.declare_parameter("camera_topic", DEFAULT_CAMERA_TOPIC)
        self.declare_parameter("description_topic", DEFAULT_DESCRIPTION_TOPIC)
        self.declare_parameter("model_path", DEFAULT_MODEL_PATH)
        self.declare_parameter("description_period_sec", 2.0)
        self.declare_parameter("max_new_tokens", 160)

        # Section: Read the resolved parameter values before creating ROS subscriptions and publishers.
        self.camera_topic = self._required_string_parameter("camera_topic")
        self.description_topic = self._required_string_parameter("description_topic")
        self.model_path = self._required_string_parameter("model_path")
        self.description_period_sec = self._positive_float_parameter("description_period_sec")
        self.max_new_tokens = self._positive_int_parameter("max_new_tokens")

        # Section: Keep the newest converted RGB frame behind a short-lived lock.
        self.bridge = CvBridge()
        self.latest_frame: Any | None = None
        self.frame_lock = Lock()

        # Section: Prevent concurrent model generations from competing for GPU memory.
        self.inference_running = False
        self.inference_lock = Lock()

        # Section: Load the processor before the model so the multimodal chat template is available.
        self.get_logger().info(f"Loading scene-description processor from: {self.model_path}")
        self.processor = AutoProcessor.from_pretrained(self.model_path)

        # Section: Select 4-bit NF4 only when CUDA is available; keep CPU startup usable for testing.
        model_kwargs: dict[str, Any] = {
            "device_map": "auto",
            "low_cpu_mem_usage": True,
        }
        if torch.cuda.is_available():
            # Section: Use BF16 when supported and FP16 otherwise for efficient CUDA inference.
            compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            model_kwargs["torch_dtype"] = compute_dtype
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )
        else:
            # Section: Use full-precision CPU loading because bitsandbytes 4-bit is CUDA-oriented here.
            model_kwargs["torch_dtype"] = torch.float32
            self.get_logger().warn("CUDA is unavailable; loading the scene model without 4-bit quantization.")

        # Section: Load the model once because every timer cycle reuses the same model instance.
        self.get_logger().info("Loading scene-description model...")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_path,
            **model_kwargs,
        )

        # Section: Disable training-only behavior because this node performs inference only.
        self.model.eval()

        # Section: Subscribe to the requested camera stream with depth one so stale frames are discarded.
        self.camera_subscription = self.create_subscription(
            RosImage,
            self.camera_topic,
            self.image_callback,
            1,
        )

        # Section: Publish one plain-text description after each completed inference cycle.
        self.description_publisher = self.create_publisher(
            String,
            self.description_topic,
            10,
        )

        # Section: Schedule non-blocking worker startup at the configured description period.
        self.description_timer = self.create_timer(
            self.description_period_sec,
            self.timer_callback,
        )

        # Section: Make the active ROS interfaces visible in startup logs.
        self.get_logger().info(f"Subscribed to camera topic: {self.camera_topic}")
        self.get_logger().info(f"Publishing scene descriptions on: {self.description_topic}")
        self.get_logger().info(
            f"Generating one description every {self.description_period_sec:.2f} seconds."
        )

    def _required_string_parameter(self, parameter_name: str) -> str:
        """Special function: Read one required non-empty ROS string parameter."""
        # Section: Read the declared parameter as a ROS string value.
        value = self.get_parameter(parameter_name).get_parameter_value().string_value

        # Section: Fail early when a required topic or model path was configured as empty.
        if not value.strip():
            raise ValueError(f"ROS parameter '{parameter_name}' must be a non-empty string.")

        # Section: Return the trimmed value for stable logging and resource lookup.
        return value.strip()

    def _positive_float_parameter(self, parameter_name: str) -> float:
        """Special function: Read and validate a positive floating-point ROS parameter."""
        # Section: Read the declared parameter as a ROS double value.
        value = self.get_parameter(parameter_name).get_parameter_value().double_value

        # Section: Reject zero, negative, NaN, and infinite timer periods.
        if value <= 0.0 or not math.isfinite(value):
            raise ValueError(f"ROS parameter '{parameter_name}' must be a finite positive number.")

        # Section: Return the validated timer period.
        return float(value)

    def _positive_int_parameter(self, parameter_name: str) -> int:
        """Special function: Read and validate a positive integer ROS parameter."""
        # Section: Read the declared parameter as a ROS integer value.
        value = self.get_parameter(parameter_name).get_parameter_value().integer_value

        # Section: Reject non-positive generation limits that would produce no useful output.
        if value <= 0:
            raise ValueError(f"ROS parameter '{parameter_name}' must be a positive integer.")

        # Section: Return the validated generation limit.
        return int(value)

    def image_callback(self, message: RosImage) -> None:
        """Special function: Convert each colour frame and retain only the newest RGB image."""
        # Section: Ignore unsupported encodings instead of interpreting depth or mono data as colour.
        if message.encoding.casefold() not in SUPPORTED_COLOR_ENCODINGS:
            self.get_logger().warn(
                f"Ignoring unsupported image encoding '{message.encoding}'; "
                "expected rgb8, bgr8, rgba8, or bgra8."
            )
            return

        try:
            # Section: Convert the ROS message to a consistent RGB array for PIL and Qwen3-VL.
            frame_rgb = self.bridge.imgmsg_to_cv2(message, desired_encoding="rgb8")

            # Section: Copy the ROS-owned buffer before the callback returns.
            with self.frame_lock:
                self.latest_frame = frame_rgb.copy()
        except Exception as exc:
            # Section: Keep the previous valid frame when one malformed message arrives.
            self.get_logger().error(f"Camera image conversion failed: {exc}")

    def timer_callback(self) -> None:
        """Special function: Snapshot the latest frame and start one background description job."""
        # Section: Reserve the model so timer callbacks cannot launch overlapping generations.
        with self.inference_lock:
            if self.inference_running:
                # Section: Skip this cycle while the previous model generation is still running.
                return

            # Section: Copy the newest frame while holding the short-lived frame lock.
            with self.frame_lock:
                if self.latest_frame is None:
                    # Section: Wait for the first camera message before attempting inference.
                    self.get_logger().warn("No camera frame has been received yet.")
                    return
                frame_rgb = self.latest_frame.copy()

            # Section: Mark inference busy before creating the worker thread.
            self.inference_running = True

        try:
            # Section: Keep model generation out of the ROS executor callback thread.
            Thread(
                target=self.describe_frame,
                args=(frame_rgb,),
                daemon=True,
            ).start()
        except Exception as exc:
            # Section: Release the busy flag if worker creation fails unexpectedly.
            with self.inference_lock:
                self.inference_running = False
            self.get_logger().error(f"Could not start scene-description worker: {exc}")

    def describe_frame(self, frame_rgb: Any) -> None:
        """Special function: Generate and publish one plain-text description for one RGB frame."""
        try:
            # Section: Convert the copied NumPy RGB array into the PIL image expected by Qwen3-VL.
            pil_image = Image.fromarray(frame_rgb)

            # Section: Build a multimodal user message with one image and the fixed scene contract.
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": pil_image,
                            "resized_height": 448,
                            "resized_width": 448,
                        },
                        {
                            "type": "text",
                            "text": SCENE_DESCRIPTION_PROMPT,
                        },
                    ],
                }
            ]

            # Section: Apply the model chat template and return tensor inputs for generation.
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )

            # Section: Move encoded tensors to the device selected by the model loader.
            inputs = inputs.to(self.model.device)

            # Section: Generate deterministic text so repeated views do not introduce sampling noise.
            with torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )

            # Section: Remove the prompt tokens before decoding only the new model response.
            generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
            description = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

            # Section: Do not publish an empty result when the model emits only special tokens.
            if not description:
                raise RuntimeError("The model returned an empty scene description.")

            # Section: Publish the generated description as a standard ROS String message.
            result_message = String()
            result_message.data = description
            self.description_publisher.publish(result_message)

            # Section: Mirror the published output in the ROS log for quick terminal inspection.
            self.get_logger().info(f"SCENE DESCRIPTION: {description}")
        except Exception as exc:
            # Section: Report inference failures without terminating the camera node.
            self.get_logger().error(f"Scene-description inference failed: {exc}")
        finally:
            # Section: Always allow a later timer cycle after success or failure.
            with self.inference_lock:
                self.inference_running = False


def main(args: list[str] | None = None) -> None:
    """Special function: Initialize ROS, run the scene node, and shut it down cleanly."""
    # Section: Initialize ROS and allow normal ROS command-line argument parsing.
    rclpy.init(args=args)

    # Section: Keep the node reference so cleanup can happen after Ctrl+C or another shutdown request.
    node: SceneDescriptionNode | None = None

    try:
        # Section: Construct the model-backed node before entering the ROS executor loop.
        node = SceneDescriptionNode()

        # Section: Process camera callbacks and periodic inference scheduling until shutdown.
        rclpy.spin(node)
    except KeyboardInterrupt:
        # Section: Treat Ctrl+C as the normal interactive shutdown path.
        print("\nCtrl+C received. Shutting down scene-description node...")
    finally:
        # Section: Destroy the node only when construction completed successfully.
        if node is not None:
            node.destroy_node()

        # Section: Avoid calling shutdown twice when ROS has already stopped.
        if rclpy.ok():
            rclpy.shutdown()


# Section: Run the ROS node only when this file is executed directly.
if __name__ == "__main__":
    # Section: Start the subscriber, model, timer, and text publisher.
    main()
