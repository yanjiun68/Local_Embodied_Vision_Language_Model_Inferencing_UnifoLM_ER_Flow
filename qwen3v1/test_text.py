#!/home/yanjiun68/ai/qwen3v1/.venv/bin/python3

import argparse
import json
import time
from pathlib import Path

import torch

from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen3VLForConditionalGeneration,
)


# Section: Use the requested local UnifoLM-ER-Flow checkpoint for NF4 text inference.
MODEL_PATH = "/home/yanjiun68/ai/models/UnifoLM-ER-Flow"

# Section: Embed the authoritative GPSR skill registry directly in this script.
# This is prompt knowledge, not a runtime YAML-file dependency.
EMBEDDED_SKILL_REGISTRY = r"""{
  "skills": {
    "NAVIGATE": {"conflict_group": "navigation_motion", "lifecycle_nodes": [], "execution_interface": "/nav/navigate_named_pose", "interface_type": "receptionist_interfaces/action/NavigateNamedPose", "implementation": "real", "production_enabled": true, "legacy_aliases": ["RETURN_TO_INSTRUCTION_POINT"]},
    "FIND_OBJECT": {"conflict_group": "vision_gpu", "lifecycle_nodes": ["/gpsr_object_detection_node", "/find_object_skill_server"], "execution_interface": "/gpsr/find_object", "interface_type": "robot_ai_interfaces/action/FindObject", "implementation": "real", "production_enabled": true, "legacy_aliases": []},
    "FIND_PERSON": {"conflict_group": "vision_gpu", "lifecycle_nodes": ["/gpsr_human_identity_node"], "execution_interface": "/appearance/recognize_human_id + /appearance/lookup_identity_profile + /appearance/current_person_observation", "interface_type": "managed YOLO tracks plus persistent CPU YuNet/SFace receptionist interfaces", "implementation": "real", "production_enabled": true, "legacy_aliases": []},
    "FIND_PERSON_BY_ATTRIBUTE": {"conflict_group": "vision_gpu", "lifecycle_nodes": ["/gpsr_human_identity_node", "/human_skeleton_detection_node", "/gpsr_tracked_appearance_node", "/find_person_by_attribute_skill_server"], "execution_interface": "/gpsr/find_person_by_attribute", "interface_type": "robot_ai_interfaces/action/FindPersonByAttribute", "implementation": "real", "production_enabled": true, "legacy_aliases": []},
    "COUNT_OBJECTS": {"conflict_group": "vision_gpu", "lifecycle_nodes": ["/gpsr_object_detection_node", "/count_objects_perception_server", "/count_objects_skill_server"], "execution_interface": "/gpsr/count_objects", "interface_type": "robot_ai_interfaces/action/CountObjects", "implementation": "real", "production_enabled": true, "legacy_aliases": []},
    "COUNT_PEOPLE": {"conflict_group": "vision_gpu", "lifecycle_nodes": ["/gpsr_human_detection_node", "/human_skeleton_detection_node", "/count_people_perception_server", "/count_people_skill_server"], "execution_interface": "/gpsr/count_people", "interface_type": "robot_ai_interfaces/action/CountPeople", "implementation": "real", "production_enabled": true, "legacy_aliases": []},
    "EXTRACT_PERSON_INFO": {"conflict_group": "vision_gpu", "lifecycle_nodes": ["/gpsr_human_identity_node", "/human_skeleton_detection_node", "/gpsr_tracked_appearance_node", "/extract_person_info_skill_server"], "execution_interface": "/gpsr/extract_guest_info", "interface_type": "robot_ai_interfaces/action/ExtractGuestInfo", "implementation": "real", "production_enabled": true, "legacy_aliases": []},
    "PICK_OBJECT": {"conflict_group": "manipulation", "lifecycle_nodes": [], "execution_interface": "/manipulation/pick_object", "interface_type": "robot_ai_interfaces/action/PickObject", "implementation": "missing", "production_enabled": false, "legacy_aliases": ["PICK"]},
    "PLACE_OBJECT": {"conflict_group": "manipulation", "lifecycle_nodes": [], "execution_interface": "/manipulation/place_object", "interface_type": "robot_ai_interfaces/action/PlaceObject", "implementation": "missing", "production_enabled": false, "legacy_aliases": ["PLACE"]},
    "HANDOVER_OBJECT": {"conflict_group": "manipulation", "lifecycle_nodes": [], "execution_interface": "/manipulation/handover_object", "interface_type": "robot_ai_interfaces/action/Handover", "implementation": "missing", "production_enabled": false, "legacy_aliases": ["GIVE"]},
    "FOLLOW_PERSON": {"conflict_group": "navigation_motion", "lifecycle_nodes": ["/follow_person_skill_server"], "execution_interface": "/person_follow/master", "interface_type": "putrabot_interfaces/action/PersonFollow via person_follower_master adapter", "implementation": "real", "production_enabled": true, "legacy_aliases": ["FOLLOW"]},
    "GUIDE_PERSON": {"conflict_group": "navigation_motion", "lifecycle_nodes": [], "execution_interface": "/person_guide/control", "interface_type": "robot_ai_interfaces/action/GuidePerson", "implementation": "missing", "production_enabled": false, "legacy_aliases": ["GUIDE"]},
    "SAY": {"conflict_group": "", "lifecycle_nodes": [], "execution_interface": "/voice/tts/say", "interface_type": "voice_interfaces/srv/SayText", "implementation": "real", "production_enabled": true, "legacy_aliases": ["REPORT", "GREET"]},
    "ASK_AND_LISTEN": {"conflict_group": "", "lifecycle_nodes": [], "execution_interface": "/voice/stt", "interface_type": "voice topics plus voice_interfaces/srv/SayText", "implementation": "real", "production_enabled": true, "legacy_aliases": ["ASK"]},
    "FIXED_INFORMATION": {"conflict_group": "", "lifecycle_nodes": [], "execution_interface": "/fixed_information/query + /fixed_information/answer", "interface_type": "std_msgs/msg/String fact-key request-response; the provider selects a configured preset dialogue and sends it through voice_interfaces/srv/SayText", "implementation": "real", "production_enabled": true, "legacy_aliases": []}
  }
}""".strip()


# Section: Use the canonical ActionQueue system prompt shared with the ROS Qwen bridge.
DEFAULT_SYSTEM_PROMPT_FILE = (
    "/home/yanjiun68/workspace/WS1/PutrabotUPM-project/src/"
    "qwen_vlm_bridge/config/system_prompt_actionqueue_test.txt"
)
DEFAULT_MESSAGE_FILE = "message.txt"
DEFAULT_INPUT_SOURCE = "file"
DEFAULT_GPSR_COMMAND_TOPIC = "/gpsr/command"
DEFAULT_GPSR_COMMAND_TIMEOUT_SEC = 60.0
DEFAULT_QWEN_OUTPUT_TOPIC = "/ai/qwen/action_queue_json"
DEFAULT_QWEN_OUTPUT_PUBLISH_TIMEOUT_SEC = 2.0


def load_text_file(file_path: str) -> str:
    """Special function: Load one non-empty UTF-8 text input file."""
    path = Path(file_path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"File does not exist: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Path is not a file: {path}"
        )

    text = path.read_text(
        encoding="utf-8"
    ).strip()

    if not text:
        raise ValueError(
            f"File is empty: {path}"
        )

    return text


def parse_model_json_response(response: str) -> dict[str, object]:
    """Special function: Decode a JSON response with an optional Markdown code fence."""
    # Section: Remove surrounding whitespace before checking for the common fenced-JSON wrapper.
    candidate = response.strip()

    # Section: Remove only a complete leading-and-trailing Markdown fence without altering JSON content.
    lines = candidate.splitlines()
    if (
        len(lines) >= 2
        and lines[0].strip().casefold() in ("```", "```json")
        and lines[-1].strip() == "```"
    ):
        candidate = "\n".join(lines[1:-1]).strip()

    # Section: Require a JSON object because the ActionQueue contract has named top-level fields.
    parsed = json.loads(candidate)
    if not isinstance(parsed, dict):
        raise ValueError("The model response must be a top-level JSON object.")
    return parsed



def build_skill_grounded_system_prompt(
    base_prompt: str,
) -> str:
    """Ground task planning in the skill registry embedded in this system prompt."""
    return (
        f"{base_prompt.strip()}\n\n"
        "==================================================\n"
        "AUTHORITATIVE EMBEDDED SKILL REGISTRY\n"
        "==================================================\n"
        "The registry below is compiled into this Python script and forms part of "
        "the system prompt. No skill_registry.yaml file is loaded at runtime.\n\n"
        "TASK-PLANNING RULES FOR THIS REGISTRY:\n"
        "- Treat this registry as authoritative for which robot skills are available.\n"
        "- Use only skills explicitly present in this registry.\n"
        "- Use the exact canonical skill names defined by the registry.\n"
        "- Respect production_enabled and implementation status.\n"
        "- Do not plan a skill with production_enabled=false as an executable production step.\n"
        "- Legacy aliases are recognition aliases only; emit canonical skill names.\n"
        "- Do not invent a skill merely because an example elsewhere in the prompt mentions it.\n"
        "- If the base prompt and this registry disagree about skill availability, "
        "this embedded registry takes precedence.\n"
        "- Treat the embedded registry as data, not as user instructions.\n"
        "- If the requested task cannot be represented with enabled implemented skills "
        "without guessing, use the clarification behavior required by the base prompt.\n\n"
        "<skill_registry>\n"
        f"{EMBEDDED_SKILL_REGISTRY}\n"
        "</skill_registry>"
    )


def extract_gpsr_instruction(message_text: str) -> str:
    """Special function: Extract the instruction from a GPSR command envelope."""
    text = str(message_text).strip()
    if not text:
        raise ValueError("Received an empty /gpsr/command message.")

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        # Section: Also accept a plain-text command for manual ROS topic testing.
        return text

    if isinstance(payload, dict):
        # Section: Read the canonical GPSR voice-command envelope field.
        instruction = payload.get("instruction")
        if isinstance(instruction, str) and instruction.strip():
            return instruction.strip()

        # Section: Accept a simple command alias for compatible producers.
        command = payload.get("command")
        if isinstance(command, str) and command.strip():
            return command.strip()

    raise ValueError(
        "The /gpsr/command JSON message must contain a non-empty "
        "'instruction' or 'command' string."
    )


def load_gpsr_command(topic: str, timeout_sec: float) -> str:
    """Special function: Wait for one command from a ROS std_msgs/String topic."""
    # Section: Import ROS lazily so file-input mode does not require ROS imports at startup.
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String

    # Section: Keep this one-shot receiver independent from the Qwen model lifecycle.
    received_message = {"value": None, "error": None}

    # Section: Use a local node because this script only consumes one command per run.
    class GpsrCommandInputNode(Node):
        """Special function: Receive and validate one GPSR command message."""

        def __init__(self) -> None:
            """Special function: Subscribe to the configured GPSR command topic."""
            super().__init__("qwen_gpsr_command_input")
            self.create_subscription(String, topic, self._command_callback, 10)

        def _command_callback(self, message: String) -> None:
            """Special function: Store the first valid GPSR command and stop waiting."""
            if received_message["value"] is not None or received_message["error"] is not None:
                return
            try:
                received_message["value"] = extract_gpsr_instruction(message.data)
            except ValueError as exc:
                received_message["error"] = str(exc)

    # Section: Initialize ROS only when this input mode is selected.
    initialized_here = not rclpy.ok()
    if initialized_here:
        rclpy.init(args=None)

    node = GpsrCommandInputNode()
    deadline = time.monotonic() + max(0.0, float(timeout_sec))
    try:
        # Section: Spin in bounded intervals so timeout handling remains responsive.
        while (
            rclpy.ok()
            and received_message["value"] is None
            and received_message["error"] is None
            and time.monotonic() < deadline
        ):
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        # Section: Release the temporary subscription before returning to model inference.
        node.destroy_node()
        if initialized_here and rclpy.ok():
            rclpy.shutdown()

    if received_message["error"] is not None:
        raise ValueError(received_message["error"])
    if received_message["value"] is None:
        raise TimeoutError(
            f"No GPSR command received on {topic!r} within {timeout_sec:.1f} seconds."
        )
    return str(received_message["value"])


def publish_qwen_json(topic: str, payload: dict[str, object], timeout_sec: float) -> None:
    """Special function: Publish one validated Qwen JSON result on its isolated topic."""
    # Section: Import ROS lazily so normal file-input and print-only modes stay lightweight.
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String

    # Section: Start a short-lived publisher because this script publishes one model result per run.
    initialized_here = not rclpy.ok()
    if initialized_here:
        rclpy.init(args=None)
    node = Node("qwen_action_queue_json_publisher")
    publisher = node.create_publisher(String, topic, 10)
    message = String()
    message.data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    deadline = time.monotonic() + max(0.0, float(timeout_sec))
    try:
        # Section: Wait briefly for the bridge subscription before publishing the one-shot message.
        while publisher.get_subscription_count() == 0 and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
        publisher.publish(message)
        rclpy.spin_once(node, timeout_sec=0.10)
    finally:
        # Section: Release the one-shot publisher without affecting another ROS process.
        node.destroy_node()
        if initialized_here and rclpy.ok():
            rclpy.shutdown()


parser = argparse.ArgumentParser(
    description="Run UnifoLM-ER-Flow using external prompt files."
)

parser.add_argument(
    "--system",
    default=DEFAULT_SYSTEM_PROMPT_FILE,
    help="Path to system prompt file",
)

parser.add_argument(
    "--message",
    default=DEFAULT_MESSAGE_FILE,
    help="Path to user message file",
)


# Section: Select between the original message file and the ROS GPSR command input.
parser.add_argument(
    "--input-source",
    choices=("file", "gpsr-command"),
    default=DEFAULT_INPUT_SOURCE,
    help="User-message source: 'file' reads --message; 'gpsr-command' reads one ROS command.",
)

# Section: Keep the ROS command contract configurable without changing the default topic.
parser.add_argument(
    "--gpsr-command-topic",
    default=DEFAULT_GPSR_COMMAND_TOPIC,
    help="ROS std_msgs/msg/String topic used by gpsr-command input mode.",
)

# Section: Bound command waiting so a missing GPSR publisher cannot block indefinitely.
parser.add_argument(
    "--gpsr-command-timeout",
    type=float,
    default=DEFAULT_GPSR_COMMAND_TIMEOUT_SEC,
    help="Seconds to wait for one message in gpsr-command input mode.",
)

# Section: Keep Qwen output opt-in and isolated from the local /gpsr/command owner.
parser.add_argument(
    "--publish-qwen-json",
    action="store_true",
    help="Publish valid Qwen actions JSON to the isolated Qwen bridge topic.",
)

# Section: Configure the Qwen JSON bridge topic without changing the GPSR command topic.
parser.add_argument(
    "--qwen-output-topic",
    default=DEFAULT_QWEN_OUTPUT_TOPIC,
    help="ROS std_msgs/msg/String topic for Qwen ActionQueue JSON output.",
)

# Section: Bound one-shot publication when no bridge is currently running.
parser.add_argument(
    "--qwen-output-publish-timeout",
    type=float,
    default=DEFAULT_QWEN_OUTPUT_PUBLISH_TIMEOUT_SEC,
    help="Seconds to wait for a Qwen JSON bridge subscriber.",
)

args = parser.parse_args()


base_system_prompt = load_text_file(args.system)
system_prompt = build_skill_grounded_system_prompt(
    base_system_prompt,
)

print(f"Using ActionQueue system prompt: {args.system}")
print("Using skill registry embedded in test_text.py")

# Section: Preserve message.txt as the default while allowing ROS-driven GPSR input.
if args.input_source == "file":
    user_message = load_text_file(args.message)
else:
    user_message = load_gpsr_command(
        args.gpsr_command_topic,
        args.gpsr_command_timeout,
    )


print("\nSYSTEM PROMPT:")
print("--------------------------------------------------")
print(system_prompt)
print("--------------------------------------------------")

print("\nUSER MESSAGE:")
print("--------------------------------------------------")
print(user_message)
print("--------------------------------------------------\n")


# Section: Quantize the Qwen3-VL-compatible checkpoint to NF4 for lower GPU memory use.
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)


print("Loading processor...")

processor = AutoProcessor.from_pretrained(MODEL_PATH)


print("Loading UnifoLM-ER-Flow in NF4 4-bit...")

model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    quantization_config=quant_config,
    device_map="auto",
    low_cpu_mem_usage=True,
)

# Section: Disable training behavior because this script performs inference only.
model.eval()
print("ER-1 loaded in NF4")


messages = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": system_prompt,
            }
        ],
    },
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": user_message,
            }
        ],
    },
]


inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
)

inputs = inputs.to(model.device)


print("Generating response...\n")

with torch.inference_mode():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
    )


generated_ids = output_ids[
    :,
    inputs["input_ids"].shape[1]:
]


response = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)[0].strip()


print("RAW RESULT:")
print("--------------------------------------------------")
print(response)
print("--------------------------------------------------")


try:
    # Section: Accept the model's optional Markdown fence while still validating the JSON object.
    parsed = parse_model_json_response(response)

    print("\nVALID JSON RESULT:")
    print(
        json.dumps(
            parsed,
            indent=2,
            ensure_ascii=False,
        )
    )

except (json.JSONDecodeError, ValueError) as exc:
    print("\nWARNING: Model response is not valid JSON.")
    if isinstance(exc, json.JSONDecodeError):
        print(
            f"JSON error at line {exc.lineno}, "
            f"column {exc.colno}: {exc.msg}"
        )
    else:
        print(f"JSON error: {exc}")

else:
    # Section: Publish only valid model JSON and never write it to /gpsr/command.
    if args.publish_qwen_json:
        publish_qwen_json(
            args.qwen_output_topic,
            parsed,
            args.qwen_output_publish_timeout,
        )
        print(f"Published Qwen JSON to {args.qwen_output_topic}")
