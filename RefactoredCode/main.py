from typing import Dict, Any, List, Optional
from autobahn.twisted.component import Component, run
from twisted.internet.defer import inlineCallbacks

from config import WAMP_REALM
from prompts import SCENARIOS
from profile_manager import ProfileManager
from robot_controller import RobotController


def terminal_setup() -> Dict[str, Any]:
    ProfileManager.ensure_dirs()

    print("\n" + "=" * 56)
    print("  APHASIA ROBOT SESSION SETUP")
    print("=" * 56)

    while True:
        raw = input("\nEnter participant first and last name: ").strip()
        parts = raw.split()
        if len(parts) >= 2:
            break
        print("  Please enter both a first name and a last name.")

    pid = ProfileManager.make_participant_id(parts[0], parts[-1])
    profile = ProfileManager.load(pid)

    if not profile.get("preferred_name"):
        profile["preferred_name"] = parts[0].capitalize()

    next_scen = profile.get("suggested_next_scenario", {})
    if next_scen.get("scenario"):
        print(
            f"\n  Suggested today: {next_scen['scenario']} ({next_scen.get('reason', '')})")

    notes = profile.get("session_notes", {})
    if notes.get("what_was_hard"):
        print(f"  Previously hard: {', '.join(notes['what_was_hard'])}")

    print("\nSelect a scenario:")
    for key, sc in SCENARIOS.items():
        print(f"  [{key}] {sc['name']}")

    while True:
        raw_choice = input(
            "\nEnter scenario number (default 1): ").strip() or "1"
        if raw_choice in SCENARIOS:
            break
        print(f"  Please enter a number between 1 and {len(SCENARIOS)}.")

    scenario = SCENARIOS[raw_choice]
    print(f"\n  Participant : {pid}")
    print(f"  Scenario    : {scenario['name']}")
    print("=" * 56 + "\n")

    return {"participant_id": pid, "profile": profile, "scenario": scenario}


if __name__ == "__main__":
    config = terminal_setup()
    active_controller: List[Optional[RobotController]] = [None]

    wamp = Component(
        transports=[{
            "url": "ws://wamp.robotsindeklas.nl",
            "serializers": ["msgpack"],
            "max_retries": 0,
        }],
        realm=WAMP_REALM,
    )

    @inlineCallbacks
    def on_join(session, details):
        controller = RobotController(session, config)
        active_controller[0] = controller
        yield controller.start()

    wamp.on_join(on_join)

    try:
        run([wamp])
    except KeyboardInterrupt:
        print("\n Keyboard interrupt. Saving session before exit...")
        if active_controller[0] is not None:
            active_controller[0].exit_reason = "keyboard_interrupt"
            active_controller[0]._stop_watchdog()
            active_controller[0].save_session_log()
        print(" Done.")
