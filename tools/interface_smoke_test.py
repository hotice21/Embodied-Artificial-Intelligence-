import os
import sys

import offshoot


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
GAME_ROOT = os.environ.get(
    "ISAAC_GAME_ROOT",
    os.path.join(PROJECT_ROOT, "The Binding of Isaac Rebirth Repentance")
)
MOD_PATH = os.path.join(GAME_ROOT, "mods", "szx_chinese_console_3001774454", "metadata.xml")


def check(condition, message):
    if not condition:
        raise AssertionError(message)
    print("OK - " + message)


def list_windows():
    try:
        import win32gui
    except ImportError:
        return []

    windows = []

    def collect(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if title:
                windows.append(title)
        return True

    try:
        win32gui.EnumWindows(collect, None)
    except Exception as exc:
        print("INFO - Window enumeration skipped: %s" % exc)
    return windows


def main():
    sys.path.insert(0, PROJECT_ROOT)

    games = offshoot.discover("Game")
    agents = offshoot.discover("GameAgent")

    check("SerpentIsaacGame" in games, "SerpentIsaacGame is discoverable")
    check("SerpentIsaacSystemTestGameAgent" in agents, "SerpentIsaacSystemTestGameAgent is discoverable")

    game = games["SerpentIsaacGame"]()
    check(os.path.isfile(game.kwargs["executable_path"]), "isaac-ng.exe path exists")
    check(os.path.isfile(MOD_PATH), "teammate console mod is installed")
    check("REGION_GAMEPLAY" in game.screen_regions, "REGION_GAMEPLAY screen region exists")
    check("MOVE_UP" in game.api_class.ACTION_KEY_MAP, "MOVE_UP action exists")
    check("SHOOT_RIGHT" in game.api_class.ACTION_KEY_MAP, "SHOOT_RIGHT action exists")

    agent_class = agents["SerpentIsaacSystemTestGameAgent"]
    check(hasattr(agent_class, "handle_play"), "agent handle_play method exists")
    check(hasattr(agent_class, "setup_play"), "agent setup_play method exists")

    matching_windows = [
        title for title in list_windows()
        if "Binding of Isaac" in title or "Isaac" in title
    ]

    if matching_windows:
        print("INFO - Isaac-like window title(s): " + " | ".join(matching_windows))
    else:
        print("INFO - No Isaac game window is currently open; launch the game for live window testing.")


if __name__ == "__main__":
    main()
