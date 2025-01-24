import subprocess

def get_clipboard_content() -> str:
    try:
        return subprocess.check_output(['pbpaste'], text=True)
    except subprocess.CalledProcessError:
        return ""
    except FileNotFoundError:
        raise RuntimeError("Clipboard access requires macOS with pbcopy/pbpaste")

def safe_subprocess_run(command: str) -> str:
    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        check=False
    )
    return f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
