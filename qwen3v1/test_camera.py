#!/usr/bin/env python3
"""Run command-triggered UnifoLM-ER-Flow analysis on the latest ROS camera frame."""

# Section: Use paths for the adjacent base prompt.
from pathlib import Path
from threading import Lock, Thread
from typing import Any

# Section: Convert the latest ROS RGB frame into the image type expected by UnifoLM-ER-Flow.
from PIL import Image

# Section: Receive camera images and text commands, then publish JSON and speech-ready output.
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image as RosImage
from std_msgs.msg import String

# Section: Load the local NF4-quantized UnifoLM-ER-Flow model and its Qwen3-VL-compatible chat processor.
import torch
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen3VLForConditionalGeneration,
)

# Section: Support both direct script execution and module execution from the workspace root.
try:
    from .vision_commands import (
        VisionCommand,
        VisionCommandError,
        build_vision_prompt,
        build_vision_speech,
        normalize_model_result,
        parse_markdown_model_result,
        parse_vision_command,
    )
except ImportError:
    from vision_commands import (
        VisionCommand,
        VisionCommandError,
        build_vision_prompt,
        build_vision_speech,
        normalize_model_result,
        parse_markdown_model_result,
        parse_vision_command,
    )


# Section: Use the requested local UnifoLM-ER-Flow checkpoint for camera inference.
MODEL_PATH = "/home/yanjiun68/ai/models/UnifoLM-ER-Flow"

# Section: Resolve the Markdown base prompt relative to this script instead of the launch directory.
VISION_PROMPT_FILE = Path(__file__).with_name("vision_prompt.md")

# Section: Define independent ROS interfaces for camera input, vision requests, and plain-text results.
DEFAULT_CAMERA_TOPIC = "/camera/image_raw"
DEFAULT_VISION_COMMAND_TOPIC = "/vision/command"
DEFAULT_VISION_RESULT_TOPIC = "/vision/result"

# Section: Accept only ordinary colour encodings so depth frames never reach the vision-language model.
SUPPORTED_COLOR_ENCODINGS = frozenset(("rgb8", "bgr8", "rgba8", "bgra8"))

def load_text_file(path: str | Path) -> str:
    """Special function: Load one required, non-empty UTF-8 prompt file."""
    # Section: Resolve user-relative paths before checking their existence and type.
    resolved_path = Path(path).expanduser().resolve()

    # Section: Produce useful startup errors for missing or directory prompt paths.
    if not resolved_path.exists():
        raise FileNotFoundError(f"File does not exist: {resolved_path}")
    if not resolved_path.is_file():
        raise ValueError(f"Path is not a file: {resolved_path}")

    # Section: Strip only outer whitespace so deliberate prompt formatting remains intact.
    text = resolved_path.read_text(encoding="utf-8").strip()
    if not text:
        raise RuntimeError(f"File is empty: {resolved_path}")
    return text


# Section: Identify explicitly configured depth topics before the node creates camera subscriptions.
def is_depth_camera_topic(topic: str) -> bool:
    """Special function: Return whether a ROS topic path explicitly names a depth image stream."""
    # Section: Compare individual path segments so names such as depthai do not reject a colour topic.
    return any(segment.casefold() == "depth" for segment in topic.split("/"))


class QwenCameraNode(Node):
    """Special ROS node: Answer one validated vision command from one latest camera snapshot."""

    def __init__(self) -> None:
        """Special function: Configure ROS interfaces and load the local UnifoLM-ER-Flow model once."""
        # Section: Give the node a stable graph name for ROS inspection and remapping.
        super().__init__("qwen_vl_camera_node")

        # Section: Declare all interfaces as parameters so launch files can remap them safely.
        self.declare_parameter("camera_topic", DEFAULT_CAMERA_TOPIC)
        self.declare_parameter("vision_command_topic", DEFAULT_VISION_COMMAND_TOPIC)
        self.declare_parameter("vision_result_topic", DEFAULT_VISION_RESULT_TOPIC)

        # Section: Read the resolved ROS parameter values once during node startup.
        camera_topic = self._string_parameter("camera_topic")
        command_topic = self._string_parameter("vision_command_topic")
        result_topic = self._string_parameter("vision_result_topic")

        # Section: Refuse an explicitly named depth stream because this node is for normal colour-camera testing.
        if is_depth_camera_topic(camera_topic):
            raise ValueError(
                "camera_topic must name a normal colour camera stream, not a depth topic."
            )

        # Section: Store the latest RGB frame behind a lock shared with command-triggered snapshots.
        self.bridge = CvBridge()
        self.latest_frame: Any | None = None
        self.frame_lock = Lock()

        # Section: Atomically reject overlapping GPU inference requests instead of racing worker threads.
        self.inference_running = False
        self.inference_lock = Lock()

        # Section: Load only foundational visual behavior; each command adds its own output contract.
        self.vision_prompt = load_text_file(VISION_PROMPT_FILE)

        # Section: Load the processor before the quantized model so image/text chat templates are available.
        self.get_logger().info("Loading UnifoLM-ER-Flow processor...")
        self.processor = AutoProcessor.from_pretrained(MODEL_PATH)

        # Section: Quantize the Qwen3-VL-compatible checkpoint to NF4 for lower GPU memory use.
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

        # Section: Load the local model once because each vision command reuses the same model instance.
        self.get_logger().info("Loading UnifoLM-ER-Flow in NF4 4-bit...")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            quantization_config=quant_config,
            device_map="auto",
            low_cpu_mem_usage=True,
        )

        # Section: Disable training behavior because this ROS node performs inference only.
        self.model.eval()
        self.get_logger().info("ER-1 loaded in NF4")

        # Section: Subscribe with depth one because only the newest camera frame is useful for a request.
        self.camera_subscription = self.create_subscription(
            RosImage,
            camera_topic,
            self.image_callback,
            1,
        )

        # Section: Receive a bounded text/JSON command that triggers analysis of one current snapshot.
        self.command_subscription = self.create_subscription(
            String,
            command_topic,
            self.vision_command_callback,
            10,
        )

        # Section: Publish exactly one plain-text response string for each completed vision command.
        self.result_publisher = self.create_publisher(String, result_topic, 10)

        # Section: Log the public command grammar so it is visible in normal ROS startup output.
        self.get_logger().info(f"Subscribed to camera topic: {camera_topic}")
        self.get_logger().info(f"Listening for vision commands on: {command_topic}")
        self.get_logger().info(f"Publishing vision text responses on: {result_topic}")
        self.get_logger().info(
            "Supported commands: 'count <object>', 'locate <object>', "
            "'reference <largest|smallest|features> [object]'."
        )

    def _string_parameter(self, parameter_name: str) -> str:
        """Special function: Read one declared ROS string parameter and reject empty required topics."""
        # Section: Convert the ROS parameter value to text without allowing non-string topic values.
        value = self.get_parameter(parameter_name).get_parameter_value().string_value

        if not value:
            raise ValueError(f"ROS parameter '{parameter_name}' must be a non-empty string.")
        return value

    def image_callback(self, message: RosImage) -> None:
        """Special function: Convert and retain the most recent camera image as an RGB snapshot source."""
        try:
            # Section: Reject depth and mono encodings before CvBridge can transform them into misleading RGB data.
            encoding = message.encoding.casefold()
            if encoding not in SUPPORTED_COLOR_ENCODINGS:
                self.get_logger().warn(
                    "Ignored non-colour camera frame with encoding "
                    f"'{message.encoding}'; only normal colour images are supported."
                )
                return

            # Section: Ask CvBridge for RGB so PIL and the UnifoLM processor receive consistent channels.
            frame_rgb = self.bridge.imgmsg_to_cv2(
                message,
                desired_encoding="rgb8",
            )

            # Section: Copy the ROS-owned image buffer before the callback returns.
            with self.frame_lock:
                self.latest_frame = frame_rgb.copy()
        except Exception as exc:
            # Section: Keep the previous valid frame when one malformed image message arrives.
            self.get_logger().error(f"Image conversion failed: {exc}")

    def vision_command_callback(self, message: String) -> None:
        """Special function: Validate one external command then start analysis of one latest frame."""
        try:
            # Section: Parse JSON or concise plaintext into an allowlisted canonical vision command.
            command = parse_vision_command(message.data)
        except VisionCommandError as exc:
            # Section: Return one plain-language invalid-command response rather than silently dropping input.
            self.get_logger().warn(f"Rejected vision command: {exc}")
            self.publish_text_response(build_vision_speech(None, "invalid_command"))
            return

        # Section: Log canonical data only, keeping raw external command content out of model diagnostics.
        self.get_logger().info(
            "Received vision command: "
            f"operation={command.operation}, object={command.target_object}, "
            f"reference={command.reference}"
        )
        self.request_analysis(command)

    def request_analysis(self, command: VisionCommand) -> None:
        """Special function: Snapshot the current frame and start exactly one non-blocking inference worker."""
        # Section: Reserve the model before copying frames so concurrent commands cannot start two workers.
        with self.inference_lock:
            if self.inference_running:
                self.get_logger().warn("Rejected command because UnifoLM-ER-Flow inference is already running.")
                self.publish_text_response(build_vision_speech(command, "busy"))
                return

            # Section: Copy a single current camera image while holding its short-lived frame lock.
            with self.frame_lock:
                if self.latest_frame is None:
                    self.get_logger().warn("Rejected command because no camera image is available.")
                    self.publish_text_response(build_vision_speech(command, "no_frame"))
                    return
            frame_rgb = self.latest_frame.copy()

            # Section: Set this before spawning so a fast second callback cannot race the worker startup.
            self.inference_running = True

        try:
            # Section: Keep expensive model generation off the ROS executor thread.
            Thread(
                target=self.analyze_frame,
                args=(frame_rgb, command),
                daemon=True,
            ).start()
        except Exception as exc:
            # Section: Release the reservation and report a local failure if thread startup ever fails.
            with self.inference_lock:
                self.inference_running = False
            self.get_logger().error(f"Could not start UnifoLM-ER-Flow inference: {exc}")
            self.publish_text_response(build_vision_speech(command, "inference_error"))

    def analyze_frame(
        self,
        frame_rgb: Any,
        command: VisionCommand,
    ) -> None:
        """Special function: Run UnifoLM-ER-Flow on one image/command pair and publish its validated response."""
        try:
            # Section: Convert the copied RGB ndarray into the in-memory PIL image UnifoLM-ER-Flow expects.
            pil_image = Image.fromarray(frame_rgb)

            # Section: Build a command-specific prompt from safe canonical fields and a fixed Markdown contract.
            command_prompt = build_vision_prompt(self.vision_prompt, command)
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
                            "text": command_prompt,
                        },
                    ],
                }
            ]

            # Section: Apply the model chat template and move all encoded inputs to the model device.
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device)

            # Section: Use deterministic decoding because this is a structured perception response.
            with torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                )

            # Section: Remove the prompt tokens before decoding only the generated model response.
            generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
            model_text = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

            # Section: Preserve special tokens in diagnostic logs so immediate EOS output is identifiable.
            raw_model_text = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )[0]

            # Section: Fail explicitly when decoding removed all generated special tokens or whitespace.
            if not model_text:
                self.get_logger().error(
                    f"UnifoLM-ER-Flow returned an empty response; raw output: {raw_model_text!r}"
                )
                raise RuntimeError("The model returned an empty response.")

            # Section: Parse labelled Markdown defensively before strict semantic validation and ROS publication.
            raw_result = parse_markdown_model_result(model_text)
            normalized = normalize_model_result(raw_result, command)
            # Section: Publish one speech-ready string without JSON fields or geometry data.
            self.publish_text_response(
                build_vision_speech(
                    command,
                    normalized["status"],
                    normalized["result"],
                )
            )
        except Exception as exc:
            # Section: Return one plain-language failure result while retaining detailed diagnostics in the log.
            self.get_logger().error(f"UnifoLM-ER-Flow inference failed: {exc}")
            self.publish_text_response(build_vision_speech(command, "model_error"))
        finally:
            # Section: Always allow the next command after model, Markdown, or publication failure.
            with self.inference_lock:
                self.inference_running = False

    def publish_text_response(
        self,
        response_text: str,
    ) -> None:
        """Special function: Publish and log one required plain-text vision response."""
        # Section: Publish only the response string on the sole vision-result topic.
        result_message = String()
        result_message.data = response_text
        self.result_publisher.publish(result_message)

        # Section: Keep terminal diagnostics equally concise and identical to the published string.
        self.get_logger().info(f"VISION RESPONSE: {response_text}")


def main(args: list[str] | None = None) -> None:
    """Special function: Initialize ROS, run the command-driven node, and cleanly release resources."""
    # Section: Let rclpy parse standard ROS arguments and parameter overrides.
    rclpy.init(args=args)
    node: QwenCameraNode | None = None

    try:
        # Section: Construct the model node before spinning so subscriptions exist for command publishers.
        node = QwenCameraNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        # Section: Treat Ctrl+C as the normal interactive shutdown path.
        print("\nCtrl+C received. Shutting down Qwen camera node...")
    finally:
        # Section: Destroy only a successfully created node and avoid calling a nonexistent node.shutdown().
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


# Section: Run the ROS node only when this module is invoked as a script.
if __name__ == "__main__":
    main()
