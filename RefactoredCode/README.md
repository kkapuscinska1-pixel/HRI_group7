# Introductory Conversation Robot

A verbal interaction system that conducts therapeutic, personalized conversations with a Person with Aphasia (PwA) using a social robot, GPT-4o.

---

## Requirements

* Python 3.9 or higher
* A robot connected via the [Robots in de Klas](https://portal.robotsindeklas.nl/) platform
* An OpenAI API key

---

## Installation

1. **Clone or download** this repository.
2. **Install dependencies**:
```bash
pip install -r requirements.txt

```


3. **Set your OpenAI API key** as an environment variable (e.g., `export OPENAI_API_KEY="your-key-here"` on macOS/Linux).

---

## Configuration

Most behavioral settings, timeouts, and pacing thresholds can be tuned in **`config.py`**.

| Constant | Description |
| --- | --- |
| `WAMP_REALM` | Your robot's unique WAMP realm ID. |
| `MAX_SESSION_DURATION` | Max conversation length in seconds (default: 300). |
| `SLOW_RESPONSE_THRESHOLD` | Avg. response time (s) that triggers adaptive pacing. |

---

## Running the Program

Run the main entry point to start the session setup:

```bash
python main.py

```

1. **Setup:** The program will prompt for the participant's name to load/create their profile and ask you to select a scenario (e.g., Small Talk, Therapy Practice, Social Roleplay).
2. **Interaction:** The robot will greet the user based on their historical profile memory.
3. **Pacing:** The system will dynamically adapt to the user’s speech speed and struggle level.
4. **Conclusion:** After the session ends, the system automatically analyzes the conversation and updates the participant's profile with new insights.

---

## Key Features

* **Named Profile System:** Automatic loading/saving of participant preferences and session history.
* **Dynamic Pacing:** Automatically slows down robot response times if the user is struggling.
* **Robust Error Handling:** Includes a background watchdog and automatic retry logic for API calls.
* **Clarification Middleware:** Gently prompts for repetition if speech is unintelligible.
* **Session Logging:** Generates comprehensive JSON logs containing pacing metrics and full transcripts.

---

## Project Structure

```
.
├── main.py               # Entry point and terminal setup
├── robot_controller.py   # WAMP session state machine and robot behavior
├── llm_manager.py        # OpenAI API interaction and JSON parsing
├── profile_manager.py    # Profile I/O and data merging logic
├── prompts.py            # System prompts and scenario definitions
├── config.py             # Global constants and tunable thresholds
├── profiles/             # Stored participant JSON profiles
└── sessions/             # Auto-saved session logs

```

---

## Stopping the Program

* **Natural End:** Say a farewell phrase like "goodbye" or "that's all."
* **Emergency:** Press `Ctrl+C` to stop; the system will safely save your progress before exiting.