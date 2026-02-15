import subprocess
import sys


def get_working_requirements():
    """Extract EXACT versions from current working venv"""
    result = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True
    )

    if result.returncode == 0:
        requirements = result.stdout.strip()
        print("=== YOUR WORKING DEPENDENCIES ===")
        print(requirements)

        # Save to file
        with open("requirements-working.txt", "w") as f:
            f.write(requirements)
        print("\n✅ Saved to requirements-working.txt")
    else:
        print("❌ pip freeze failed")


if __name__ == "__main__":
    get_working_requirements()
