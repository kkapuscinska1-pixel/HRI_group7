WAMP_REALM = "rie.6a26b0028a2cba4f82b8706b"
LLM_MODEL = "gpt-4o"
LLM_TIMEOUT_SECS = 20.0        # OpenAI request hard timeout
LLM_RETRY_DELAYS = (1.0, 3.0)  # Exponential backoff for API retries

MAX_SESSION_DURATION = 60 * 5  # Seconds before time-limit farewell

PROFILES_DIR = "profiles"
SESSIONS_DIR = "sessions"

# --- Pacing ---
PAUSE_BASE = 1.0               # Always applied after robot speech
PAUSE_AUTO_SLOW_BONUS = 1.0    # Added when avg response time > threshold
PAUSE_SLOW_BONUS = 1.5         # Added when LLM signals pace="slow"
SLOW_RESPONSE_THRESHOLD = 4.0  # Seconds: triggers auto-slow pacing
POST_SPEECH_MIC_DELAY = 0.8    # Wait after TTS before opening mic

# --- STT reliability ---
STT_BUFFER_DELAY = 0.8         # Seconds to wait for more STT finals
STT_SILENCE_TIMEOUT = 25.0     # Restart STT stream if no activity

# --- WAMP calls ---
WAMP_CALL_TIMEOUT = 12.0       # Seconds before a WAMP call is cancelled

# --- Watchdog ---
WATCHDOG_INTERVAL = 5.0        # How often the watchdog checks (seconds)
WATCHDOG_FREEZE_TIMEOUT = 45.0  # How long a state may be unchanged

# --- Input ---
VALID_SHORT_WORDS = frozenset({"i", "a", "y", "n", "o"})
EXIT_PHRASES = frozenset({
    "goodbye", "bye", "quit", "exit",
    "stop", "that's all", "thats all", "no more",
})
