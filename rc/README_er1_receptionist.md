# ER-1 RoboCup Receptionist prototype

## Install

Create/activate a Python 3.10+ environment, install a CUDA-enabled PyTorch build
appropriate for your machine, then:

```bash
pip install -r requirements_er1_receptionist.txt
```

For an 8 GB RTX 4060, use `--load-in-4bit`.

## 1. Register guest 1 at the entrance

```bash
python robocup_receptionist_er1.py register \
  --image guest1.jpg \
  --introduction "Hi, my name is Alice and my favorite drink is orange juice." \
  --state receptionist_state.json \
  --load-in-4bit
```

## 2. Register guest 2

```bash
python robocup_receptionist_er1.py register \
  --image guest2.jpg \
  --introduction "I'm Robin. My favorite drink is water." \
  --state receptionist_state.json \
  --load-in-4bit
```

## 3. Analyze the living-room group

The host's name and favorite drink are expected to be known before the test.
The program identifies the host as the unique visible person who does not match
any registered guest.

```bash
python robocup_receptionist_er1.py analyze \
  --image living_room.jpg \
  --host-name Morgan \
  --host-drink coffee \
  --state receptionist_state.json \
  --output receptionist_result.json \
  --annotated receptionist_debug.jpg \
  --load-in-4bit
```

The JSON result contains:

- matched registered guests
- host status and host bounding box
- confidently unoccupied chairs
- uncertain chairs
- host-facing guest descriptions
- recommended next action

## Design notes

This is intentionally an image-level prototype rather than an rclpy node.
Keep ER-1 resident in one process for the real robot; do not reload the model
for every camera frame.

For production:
- feed introductions from ASR rather than CLI;
- capture guest registration frames from the RGB camera;
- call group analysis only when the robot reaches the living room;
- publish the chosen free-seat location to the pointing/navigation layer;
- use deterministic state-machine recovery if host/seat recognition is ambiguous.
