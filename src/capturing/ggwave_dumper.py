import os
import subprocess

def run_decoder_on_recordings():
    recordings_dir = "/home/glick/Desktop/CATpack/src/capturing/recordings/chars"
    decoder_script = "../../test/ggwave_test_decoder.py"

    for filename in sorted(os.listdir(recordings_dir)):
        if filename.endswith(".wav"):
            input("Enter when ready")
            filepath = os.path.join(recordings_dir, filename)
            print(f"\n[ToAD] Running decoder on: {filename}")
            subprocess.run(["python3", decoder_script, filepath])

if __name__ == "__main__":
    run_decoder_on_recordings()
