#!/home/yanjiun68/ai/qwen3v1/.venv/bin/python3
"""Run natural-language command-triggered UnifoLM-ER-Flow analysis on the latest ROS camera frame."""

import json
import math
import uuid
from threading import Lock, Thread
from typing import Any

from PIL import Image

import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image as RosImage
from std_msgs.msg import String

import torch
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen3VLForConditionalGeneration,
)


# Local NF4-quantized UnifoLM-ER-Flow checkpoint.
MODEL_PATH = "/home/yanjiun68/ai/models/UnifoLM-ER-Flow"


# ROS interfaces.
DEFAULT_CAMERA_TOPIC = "/astra/color/image_raw"
DEFAULT_VISION_COMMAND_TOPIC = "/vision/command"
DEFAULT_VISION_RESULT_TOPIC = "/vision/result"
DEFAULT_VISION_SPEECH_TOPIC = "/vision/speech"
DEFAULT_VISION_DEBUG_IMAGE_TOPIC = "/vision/debug/image_raw"

# Bound external requests before inserting them into the model prompt.
MAX_COMMAND_LENGTH = 2000

# Stable terminal delimiter for completed responses.
COMMAND_RESULT_FOOTER_LINE = "=" * 72

# Use a dedicated prompt for natural-language, scene-relative object referencing.
# Do not reuse vision_prompt.txt here because that file still describes the older
# validated-command / JSON-contract interface.
MODEL_IMAGE_LONG_EDGE = 672
MODEL_IMAGE_MULTIPLE = 28

SPATIAL_VISION_PROMPT = """
You are the semantic vision system of a ROS 2 service robot.

Analyze only the supplied current RGB camera image and answer the user's request
using visible evidence from that image.

OBJECT-KNOWLEDGE GROUNDING

A trusted <object_knowledge_yaml> block is embedded later in this vision prompt.

Use that YAML to:
- normalize known object names and aliases when the YAML provides them;
- understand object categories or semantic groups defined by the YAML;
- understand known/default object locations or furniture associations;
- prefer meaningful surrounding-object and furniture anchors defined by the YAML;
- distinguish whether nearby objects belong to the same or a different known group.

The YAML is semantic prior knowledge, not proof of what is visible right now.
Never claim that an object is present at its configured/default location unless
the current image supports that claim. Current visual evidence takes precedence
for observed position and visibility.

EMBEDDED OBJECT KNOWLEDGE

The following object knowledge is compiled directly into this vision prompt.
No object_knowledge.yaml file is loaded at runtime.

<object_knowledge_yaml>
# Section: Incheon 2026 RoboCup@Home objects plus four retained local targets.
# Source: https://github.com/RoboCupAtHome/Incheon2026/blob/main/objects/objects.md
objects:
  # Section: Official fabrics and toy objects.
  white_shirt:
    aliases: [white shirt]
    detector_label: white shirt
    category: fabric
    probable_locations: [bedroom, living_room]
  grey_shirt:
    aliases: [grey shirt, gray shirt]
    detector_label: grey shirt
    category: fabric
    probable_locations: [bedroom, living_room]
  blue_shirt:
    aliases: [blue shirt]
    detector_label: blue shirt
    category: fabric
    probable_locations: [bedroom, living_room]
  black_shirt:
    aliases: [black shirt]
    detector_label: black shirt
    category: fabric
    probable_locations: [bedroom, living_room]
  hand_towel:
    aliases: [hand towel]
    detector_label: hand towel
    category: fabric
    probable_locations: [kitchen, bedroom]
  rubiks_cube:
    aliases: [rubiks cube, "rubik's cube"]
    detector_label: rubiks cube
    category: toy
    probable_locations: [living_room, bedroom]

  # Section: Official snacks, fruits, and food objects.
  pringles:
    aliases: [pringles, pringles can]
    detector_label: pringles can
    category: snack
    probable_locations: [kitchen, living_room]
  seaweed:
    aliases: [seaweed]
    detector_label: seaweed
    category: snack
    probable_locations: [kitchen, dining_room]
  apple:
    aliases: [apple]
    detector_label: apple
    category: fruit
    probable_locations: [kitchen, dining_room]
  peach:
    aliases: [peach]
    detector_label: peach
    category: fruit
    probable_locations: [kitchen, dining_room]
  mangostane:
    aliases: [mangostane, mangosteen]
    detector_label: mangosteen
    category: fruit
    probable_locations: [kitchen, dining_room]
  lemon:
    aliases: [lemon]
    detector_label: lemon
    category: fruit
    probable_locations: [kitchen, dining_room]
  yellow_bellpepper:
    aliases: [yellow bellpepper, yellow bell pepper]
    detector_label: yellow bell pepper
    category: fruit
    probable_locations: [kitchen, dining_room]
  red_bellpepper:
    aliases: [red bellpepper, red bell pepper]
    detector_label: red bell pepper
    category: fruit
    probable_locations: [kitchen, dining_room]
  instant_noodles:
    aliases: [instant noodles]
    detector_label: instant noodles
    category: food
    probable_locations: [kitchen, dining_room]
  cornflakes:
    aliases: [cornflakes, corn flakes]
    detector_label: cornflakes
    category: food
    probable_locations: [kitchen, dining_room]

  # Section: Official drinks and cleaning supplies.
  coke:
    aliases: [coke, coca cola, cola]
    detector_label: coke
    category: drink
    probable_locations: [kitchen, dining_room, living_room]
  red_bull:
    aliases: [red bull, red bull can]
    detector_label: red bull can
    category: drink
    probable_locations: [kitchen, dining_room, living_room]
  milk:
    aliases: [milk, milk carton]
    detector_label: milk
    category: drink
    probable_locations: [kitchen, dining_room]
  soju:
    aliases: [soju]
    detector_label: soju bottle
    category: drink
    probable_locations: [kitchen, dining_room, living_room]
  pepsi:
    aliases: [pepsi, pepsi can]
    detector_label: pepsi can
    category: drink
    probable_locations: [kitchen, dining_room, living_room]
  dishwasher_tab:
    aliases: [dishwasher tab, dishwasher tablet]
    detector_label: dishwasher tablet
    category: cleaning_supply
    probable_locations: [kitchen]
  sponge:
    aliases: [sponge]
    detector_label: sponge
    category: cleaning_supply
    probable_locations: [kitchen]
  toothpaste:
    aliases: [toothpaste]
    detector_label: toothpaste
    category: cleaning_supply
    probable_locations: [bedroom]

  # Section: Official dishes and cutlery.
  cup:
    aliases: [cup, drinking cup, mug]
    detector_label: cup
    category: dish
    probable_locations: [kitchen, dining_room, living_room]
  spoon:
    aliases: [spoon]
    detector_label: spoon
    category: dish
    probable_locations: [kitchen, dining_room]
  plate:
    aliases: [plate]
    detector_label: plate
    category: dish
    probable_locations: [kitchen, dining_room]
  knife:
    aliases: [knife]
    detector_label: knife
    category: dish
    probable_locations: [kitchen, dining_room]
  fork:
    aliases: [fork]
    detector_label: fork
    category: dish
    probable_locations: [kitchen, dining_room]
  bowl:
    aliases: [bowl]
    detector_label: bowl
    category: dish
    probable_locations: [kitchen, dining_room]

  # Section: Preserve local object commands that predate the Incheon list.
  banana:
    aliases: [banana]
    detector_label: banana
    category: fruit
    probable_locations: [kitchen, dining_room]
  bottle:
    aliases: [bottle, water bottle]
    detector_label: bottle
    category: drink
    probable_locations: [kitchen, dining_room, living_room, bedroom]
  chair:
    aliases: [chair, seat]
    detector_label: chair
    probable_locations: [dining_room, living_room, bedroom, entrance]
  chips:
    aliases: [chips, potato chips]
    detector_label: chips
    category: snack
    probable_locations: [kitchen, living_room]

# Section: Singular canonical categories accept official plural class names as aliases.
categories:
  fabric:
    aliases: [fabrics]
    members: [white shirt, grey shirt, blue shirt, black shirt, hand_towel]
    probable_locations: [bedroom, living_room, kitchen]
  toy:
    aliases: [toys]
    members: [rubiks_cube]
    probable_locations: [living_room, bedroom]
  snack:
    aliases: [snacks]
    members: [pringles, seaweed, chips]
    probable_locations: [kitchen, dining_room, living_room]
  fruit:
    aliases: [fruits]
    members: [apple, peach, mangostane, lemon, yellow_bellpepper, red_bellpepper, banana]
    probable_locations: [kitchen, dining_room]
  food:
    aliases: [foods]
    members: [instant_noodles, cornflakes]
    probable_locations: [kitchen, dining_room]
  drink:
    aliases: [drinks]
    members: [coke, red_bull, milk, soju, pepsi, bottle]
    probable_locations: [kitchen, dining_room, living_room, bedroom]
  cleaning_supply:
    aliases: [cleaning supplies, cleaning supply]
    members: [dishwasher_tab, sponge, toothpaste]
    probable_locations: [kitchen, bedroom]
  dish:
    aliases: [dishes]
    members: [cup, spoon, plate, knife, fork, bowl]
    probable_locations: [kitchen, dining_room, living_room]

# Section: Generic targets may match every configured object.
wildcards: [object, any_object]
</object_knowledge_yaml>

OBJECT-REFERENCE POLICY

When the request asks you to find, locate, identify, distinguish, describe, point
out, compare, or refer to a target object, do not give only a coarse statement
such as "it is on the left".

Build a useful scene-relative reference for each target using this priority:

1. SUPPORT / FURNITURE ANCHOR
   State the target's relation to visible stable furniture or a support surface
   when clear, for example:
   - on the dining table
   - on the upper shelf of the cabinet
   - inside the open cupboard
   - beside the sofa
   - below the television

2. NEARBY OBJECT ANCHORS
   State at least one useful relation to another clearly visible nearby object
   when available. Prefer distinctive objects that help a person distinguish
   the target. Examples:
   - to the right of the white bowl
   - immediately left of the cereal box
   - between the cup and the plate
   - above the blue container
   - next to the red bottle

3. IMAGE-REGION FALLBACK
   Also give the approximate position in the current camera view when useful:
   upper-left, upper-center, upper-right, center-left, center, center-right,
   lower-left, lower-center, or lower-right.

Use two independent anchors when they are clearly visible and useful. Prefer a
stable furniture anchor plus a nearby object anchor.

DISAMBIGUATION

- If several similar target objects are visible, do not arbitrarily choose one.
  Distinguish them using visible attributes and spatial anchors.
- Refer to anchors by observable properties when needed, such as "the white bowl"
  or "the tall blue bottle".
- Prefer references that uniquely identify the target from the current view.
- Never use the target object itself as its own reference anchor.

SPATIAL SAFETY

- Camera left/right means left/right in the displayed camera image.
- Use "in front of" or "behind" only when the visual evidence is clear enough.
- Do not invent metric distance, map coordinates, robot coordinates, hidden
  objects, hidden furniture, or physical dimensions.
- Do not claim a relationship unless both the target and reference anchor are
  visible enough to support it.
- If an exact relation is unclear, state the uncertainty instead of guessing.
- This node performs visual analysis only. Do not claim the robot moved,
  grasped, picked, placed, followed, or manipulated anything.

ANSWER STYLE

For a target-object location request, prefer an answer shaped like:

"The <target> is <relation to furniture/support>. It is <relation to nearby
anchor> [and <relation to second anchor>]. In the camera view it is around the
<image region>."

This is a preferred semantic structure, not a rigid wording requirement.

For other visual questions such as counting or general scene description, answer
the request naturally. Keep answers concise and grounded.

Return exactly one JSON object and no Markdown fences or surrounding prose.
Use this schema:
{
  "answer": "short evidence-based answer for the user",
  "target": {
    "label": "the single visible entity targeted by the request",
    "bbox_1000": [left, top, right, bottom],
    "confidence": 0.0
  }
}

The bbox_1000 coordinates are integers on the supplied image, normalized from
0 to 1000 in both axes. left < right and top < bottom. confidence is between
0.0 and 1.0. Select one target only when the request names or clearly implies
a visible entity. Otherwise set "target" to null. Never invent a target.
""".strip()


def validate_natural_command(raw_text: str) -> str:
    """Accept ordinary text and reject only empty or unreasonably large commands."""
    command = raw_text.strip()

    if not command:
        raise ValueError("Vision command is empty.")

    if len(command) > MAX_COMMAND_LENGTH:
        raise ValueError(
            f"Vision command is too long ({len(command)} characters); "
            f"maximum is {MAX_COMMAND_LENGTH}."
        )

    return command


def build_natural_vision_prompt(
    base_prompt: str,
    command: str,
) -> str:
    """Combine the embedded spatial/object prompt with the natural-language request."""
    return (
        f"{base_prompt.strip()}\n\n"
        "<vision_request>\n"
        f"{command}\n"
        "</vision_request>"
    )


def compute_model_image_size(
    width: int,
    height: int,
    long_edge: int = MODEL_IMAGE_LONG_EDGE,
    multiple: int = MODEL_IMAGE_MULTIPLE,
) -> tuple[int, int]:
    """Resize while preserving aspect ratio and Qwen-friendly dimensions."""
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive.")

    scale = float(long_edge) / float(max(width, height))

    resized_width = max(
        multiple,
        int(round((width * scale) / multiple)) * multiple,
    )
    resized_height = max(
        multiple,
        int(round((height * scale) / multiple)) * multiple,
    )

    return resized_width, resized_height


# Section: Validate the model's normalized rectangle before it reaches any GUI.
def normalise_target_roi(target: Any) -> dict[str, Any] | None:
    """Special function: Convert one model target object into a safe ROI payload."""
    # Section: Treat omitted, malformed, or non-object targets as no selectable entity.
    if not isinstance(target, dict):
        return None

    # Section: Require a readable target label for the overlay and crop title.
    label = target.get("label")
    if not isinstance(label, str) or not label.strip():
        return None

    # Section: Accept only four finite numeric normalized coordinates.
    bbox = target.get("bbox_1000")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in bbox
    ):
        return None

    # Section: Round model coordinates while preserving the documented 0..1000 space.
    left, top, right, bottom = (int(round(value)) for value in bbox)
    if not (0 <= left < right <= 1000 and 0 <= top < bottom <= 1000):
        return None

    # Section: Normalize confidence without allowing malformed values into JSON output.
    confidence_value = target.get("confidence", 0.0)
    if (
        isinstance(confidence_value, bool)
        or not isinstance(confidence_value, (int, float))
        or not math.isfinite(float(confidence_value))
    ):
        confidence_value = 0.0
    confidence = min(1.0, max(0.0, float(confidence_value)))

    # Section: Publish a stable target object shared by the result topic and debug GUI.
    return {
        "label": label.strip(),
        "bbox_1000": [left, top, right, bottom],
        "confidence": confidence,
    }


# Section: Preserve a useful answer if a model occasionally ignores the JSON contract.
def parse_model_response(raw_response: str) -> tuple[str, dict[str, Any] | None]:
    """Special function: Extract the semantic answer and optional validated target ROI."""
    # Section: Start with compatibility behavior for legacy plain-text model responses.
    answer = raw_response.strip()
    target: dict[str, Any] | None = None

    # Section: Parse the exact JSON contract without guessing coordinates from prose.
    try:
        payload = json.loads(answer)
    except json.JSONDecodeError:
        return answer, target

    # Section: Preserve the raw answer if the decoded payload is not a response object.
    if not isinstance(payload, dict):
        return answer, target

    # Section: Use only a non-empty declared answer as the user-facing response.
    payload_answer = payload.get("answer")
    if isinstance(payload_answer, str) and payload_answer.strip():
        answer = payload_answer.strip()

    # Section: Pass validated coordinates onward only when one target was actually selected.
    target = normalise_target_roi(payload.get("target"))
    return answer, target


# Section: Use the source-camera timestamp to pair a debug image with its result message.
def header_stamp_ns(header: Any) -> int:
    """Special function: Convert a ROS message header stamp into a stable integer key."""
    # Section: Gracefully support callers without a ROS header during error handling.
    stamp = getattr(header, "stamp", None)
    seconds = getattr(stamp, "sec", 0)
    nanoseconds = getattr(stamp, "nanosec", 0)
    return int(seconds) * 1_000_000_000 + int(nanoseconds)


def build_response(
    *,
    request_id: str,
    command: str,
    status: str,
    answer: str | None = None,
    error: str | None = None,
    target: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a stable ROS JSON envelope while keeping model output natural-language."""
    response: dict[str, Any] = {
        "request_id": request_id,
        "status": status,
        "command": command,
    }

    if answer is not None:
        response["result"] = {
            "answer": answer,
        }
        if target is not None:
            response["result"]["target"] = target
        response["speech"] = answer

    if error is not None:
        response["error"] = error

    return response


class QwenCameraNode(Node):
    """Analyze the latest camera image in response to arbitrary natural-language requests."""

    def __init__(self) -> None:
        super().__init__("qwen_vl_camera_node")

        self.declare_parameter("camera_topic", DEFAULT_CAMERA_TOPIC)
        self.declare_parameter("vision_command_topic", DEFAULT_VISION_COMMAND_TOPIC)
        self.declare_parameter("vision_result_topic", DEFAULT_VISION_RESULT_TOPIC)
        self.declare_parameter("vision_speech_topic", DEFAULT_VISION_SPEECH_TOPIC)
        self.declare_parameter(
            "vision_debug_image_topic",
            DEFAULT_VISION_DEBUG_IMAGE_TOPIC,
        )

        camera_topic = self._string_parameter("camera_topic")
        command_topic = self._string_parameter("vision_command_topic")
        result_topic = self._string_parameter("vision_result_topic")
        speech_topic = self._string_parameter("vision_speech_topic")
        debug_image_topic = self._string_parameter("vision_debug_image_topic")
        self.debug_image_topic = debug_image_topic

        self.bridge = CvBridge()
        self.latest_frame: Any | None = None
        self.latest_frame_header: Any | None = None
        self.frame_lock = Lock()

        self.inference_running = False
        self.inference_lock = Lock()

        self.vision_prompt = SPATIAL_VISION_PROMPT
        self.get_logger().info(
            "Using object knowledge embedded in astra_test_camera.py"
        )
        self.get_logger().info("Loading UnifoLM-ER-Flow processor...")
        self.processor = AutoProcessor.from_pretrained(MODEL_PATH)

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

        self.get_logger().info("Loading UnifoLM-ER-Flow in NF4 4-bit...")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            quantization_config=quant_config,
            device_map="auto",
            low_cpu_mem_usage=True,
        )

        self.model.eval()
        self.get_logger().info("ER-1 loaded in NF4")

        self.camera_subscription = self.create_subscription(
            RosImage,
            camera_topic,
            self.image_callback,
            qos_profile_sensor_data,
        )

        self.command_subscription = self.create_subscription(
            String,
            command_topic,
            self.vision_command_callback,
            10,
        )

        self.result_publisher = self.create_publisher(String, result_topic, 10)
        self.debug_image_publisher = self.create_publisher(
            RosImage,
            debug_image_topic,
            10,
        )
        self.speech_publisher = (
            self.create_publisher(String, speech_topic, 10)
            if speech_topic
            else None
        )

        self.get_logger().info(f"Subscribed to camera topic: {camera_topic}")
        self.get_logger().info(
            f"Listening for natural-language vision requests on: {command_topic}"
        )
        self.get_logger().info(f"Publishing vision JSON results on: {result_topic}")
        self.get_logger().info(
            f"Publishing analyzed debug frames on: {debug_image_topic}"
        )

        if speech_topic:
            self.get_logger().info(
                f"Publishing speech-ready answers on: {speech_topic}"
            )

        self.get_logger().info(
            "Natural-language examples: "
            "'Where is the red bottle relative to the other objects?', "
            "'Find the cup and describe its position using the table and nearby items.', "
            "'Which cereal box is beside the white bowl?', "
            "'How many bottles can you see?'"
        )

    def _string_parameter(self, parameter_name: str) -> str:
        """Read one declared ROS string parameter."""
        value = (
            self.get_parameter(parameter_name)
            .get_parameter_value()
            .string_value
        )

        # Section: Preserve the documented empty value for disabling speech output.
        if not value and parameter_name == "vision_speech_topic":
            return ""

        # Section: Require all camera, command, result, and debug image topics.
        if not value:
            raise ValueError(
                f"ROS parameter '{parameter_name}' must be a non-empty string."
            )

        return value

    def image_callback(self, message: RosImage) -> None:
        """Retain the latest RGB camera image."""
        try:
            frame_rgb = self.bridge.imgmsg_to_cv2(
                message,
                desired_encoding="rgb8",
            )

            with self.frame_lock:
                self.latest_frame = frame_rgb.copy()
                self.latest_frame_header = message.header

        except Exception as exc:
            self.get_logger().error(f"Image conversion failed: {exc}")

    def vision_command_callback(self, message: String) -> None:
        """Accept ordinary text from /vision/command and trigger one image analysis."""
        request_id = uuid.uuid4().hex

        try:
            command = validate_natural_command(message.data)
        except ValueError as exc:
            response = build_response(
                request_id=request_id,
                command=message.data.strip(),
                status="invalid_command",
                error=str(exc),
            )
            self.get_logger().warning(f"Rejected vision request: {exc}")
            self.publish_response(response)
            return

        self.get_logger().info(
            f"Received natural-language vision request [{request_id}]: {command}"
        )

        self.request_analysis(
            request_id=request_id,
            command=command,
        )

    def request_analysis(
        self,
        *,
        request_id: str,
        command: str,
    ) -> None:
        """Snapshot the latest frame and start one non-blocking inference worker."""
        with self.inference_lock:
            if self.inference_running:
                response = build_response(
                    request_id=request_id,
                    command=command,
                    status="busy",
                    error="Another vision request is already being analyzed.",
                )
                self.get_logger().warning(
                    "Rejected request because UnifoLM-ER-Flow inference is already running."
                )
                self.publish_response(response)
                return

            with self.frame_lock:
                if self.latest_frame is None:
                    response = build_response(
                        request_id=request_id,
                        command=command,
                        status="no_frame",
                        error="No camera image has been received yet.",
                    )
                    self.get_logger().warning(
                        "Rejected request because no camera image is available."
                    )
                    self.publish_response(response)
                    return

                frame_rgb = self.latest_frame.copy()
                frame_header = self.latest_frame_header

            self.inference_running = True

        try:
            Thread(
                target=self.analyze_frame,
                args=(frame_rgb, frame_header, request_id, command),
                daemon=True,
            ).start()

        except Exception as exc:
            with self.inference_lock:
                self.inference_running = False

            response = build_response(
                request_id=request_id,
                command=command,
                status="inference_error",
                error=f"Could not start UnifoLM-ER-Flow inference: {exc}",
            )
            self.get_logger().error(
                f"Could not start UnifoLM-ER-Flow inference: {exc}"
            )
            self.publish_response(response)

    def analyze_frame(
        self,
        frame_rgb: Any,
        frame_header: Any,
        request_id: str,
        command: str,
    ) -> None:
        """Run ER-1 on the image and natural-language command."""
        try:
            pil_image = Image.fromarray(frame_rgb)
            resized_width, resized_height = compute_model_image_size(
                *pil_image.size
            )

            command_prompt = build_natural_vision_prompt(
                self.vision_prompt,
                command,
            )

            self.get_logger().debug(
                "Vision input resized from "
                f"{pil_image.width}x{pil_image.height} to "
                f"{resized_width}x{resized_height}"
            )

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": pil_image,
                            "resized_height": resized_height,
                            "resized_width": resized_width,
                        },
                        {
                            "type": "text",
                            "text": command_prompt,
                        },
                    ],
                }
            ]

            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device)

            with torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                )

            generated_ids = output_ids[
                :,
                inputs["input_ids"].shape[1] :,
            ]

            raw_response = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

            answer, target = parse_model_response(raw_response)

            if not answer:
                raise RuntimeError("The model returned an empty answer.")

            response = build_response(
                request_id=request_id,
                command=command,
                status="ok",
                answer=answer,
                target=target,
            )
            response["debug"] = {
                "image_topic": self.debug_image_topic,
                "frame_stamp_ns": header_stamp_ns(frame_header),
                "roi_coordinate_space": "bbox_1000",
            }
            self.publish_response(
                response,
                debug_frame_rgb=frame_rgb,
                debug_frame_header=frame_header,
            )

        except Exception as exc:
            response = build_response(
                request_id=request_id,
                command=command,
                status="model_error",
                error=f"UnifoLM-ER-Flow inference failed: {exc}",
            )
            self.get_logger().error(
                f"UnifoLM-ER-Flow inference failed: {exc}"
            )
            self.publish_response(response)

        finally:
            with self.inference_lock:
                self.inference_running = False

    def publish_response(
        self,
        response: dict[str, Any],
        debug_frame_rgb: Any | None = None,
        debug_frame_header: Any | None = None,
    ) -> None:
        """Publish machine-readable JSON and optional speech-ready text."""
        # Section: Publish the immutable analyzed frame before its matching result envelope.
        if debug_frame_rgb is not None and debug_frame_header is not None:
            debug_image = self.bridge.cv2_to_imgmsg(
                debug_frame_rgb,
                encoding="rgb8",
            )
            debug_image.header = debug_frame_header
            self.debug_image_publisher.publish(debug_image)

        # Section: Publish the structured result after the paired debug frame is available.
        result_message = String()
        result_message.data = json.dumps(
            response,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        self.result_publisher.publish(result_message)

        speech = response.get("speech")
        if (
            self.speech_publisher is not None
            and isinstance(speech, str)
            and speech
        ):
            speech_message = String()
            speech_message.data = speech
            self.speech_publisher.publish(speech_message)

        self.log_command_output(response)

    def log_command_output(self, response: dict[str, Any]) -> None:
        """Pretty-print each complete result to the terminal."""
        formatted_response = json.dumps(
            response,
            ensure_ascii=False,
            indent=2,
        )

        request_id = response.get("request_id") or "-"
        status = response.get("status") or "-"

        footer = (
            f"{COMMAND_RESULT_FOOTER_LINE}\n"
            "END VISION COMMAND RESULT "
            f"| request_id={request_id} | status={status}\n"
            f"{COMMAND_RESULT_FOOTER_LINE}"
        )

        self.get_logger().info(
            f"VISION COMMAND RESULT:\n"
            f"{formatted_response}\n"
            f"{footer}"
        )


def main(args: list[str] | None = None) -> None:
    """Initialize ROS, run the node, and cleanly release resources."""
    rclpy.init(args=args)
    node: QwenCameraNode | None = None

    try:
        node = QwenCameraNode()
        rclpy.spin(node)

    except KeyboardInterrupt:
        print(
            "\nCtrl+C received. "
            "Shutting down Qwen camera node..."
        )

    finally:
        if node is not None:
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
