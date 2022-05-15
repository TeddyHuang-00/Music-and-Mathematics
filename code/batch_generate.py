import os
import subprocess
import time

import matplotlib.pyplot as plt
import midiutil.MidiFile as midi
import numpy as np
import pandas as pd


def call_cpp_program(file_name):
    try:
        subprocess.call([f"./bin/{file_name}"])
        return True
    except FileNotFoundError:
        print("File not found.")
    except OSError:
        print("OS error.")
    except:
        print("Unexpected error.")
    return False


def generate_midi():
    with open("./music.txt", "r") as f:
        music_seq = f.read()

    music_seq = music_seq.strip().split("> ")[1]
    music_seq = " ".join(music_seq.split("\n")[1:])

    output_dir = "."
    file_name = "best.mid"

    music_seq = music_seq.strip().split("\n")
    music_seq = [
        note.strip("(").strip(")").strip().split(",")
        for x in music_seq
        for note in x.split(") (")
    ]

    # Create MIDI object
    md_file = midi.MIDIFile(1)
    track = 0
    time_idx = 0
    md_file.addTrackName(track, time_idx, "Track 0")
    md_file.addTempo(track, time_idx, 120)
    channel = 0
    volume = 100

    for note in music_seq:
        # Add note.
        md_file.addNote(
            track=track,
            channel=channel,
            pitch=int(note[0]),
            time=time_idx,
            duration=int(note[1]) / 4,
            volume=volume,
        )
        time_idx += int(note[1]) / 4

    # Write to disk
    with open(os.path.join(output_dir, file_name), "wb") as f:
        md_file.writeFile(f)


def generate_plot():
    df: pd.DataFrame = pd.read_csv("./fitness_data.csv", header=None)
    df.dropna(axis=1, inplace=True)
    df.dropna(axis=0, inplace=True)
    data: np.ndarray = df.to_numpy()
    fig = plt.figure(figsize=(10, 5), dpi=300)
    plt.plot(data, alpha=0.1, color="tab:blue")
    plt.plot(data.mean(axis=1), color="tab:blue", label="Population")
    plt.plot(data.mean(axis=1), color="tab:red", label="Mean")
    plt.plot(data[:, 0], color="tab:green", label="Best")
    plt.xlabel("Generation #")
    plt.ylabel("Fitness Value")
    plt.title("Fitness over generations")
    plt.legend(loc="lower right")
    plt.savefig("fitness_over_generations.png")
    plt.clf()
    plt.close(fig)


def main(task_name: str):
    time_stamp = time.strftime(r"%Y%m%d-%H%M%S")
    if not os.path.isdir(os.path.join("./generated", task_name)):
        os.mkdir(os.path.join("./generated", task_name))
    if call_cpp_program(task_name):
        generate_midi()
        generate_plot()
        for fileName in [
            "best.mid",
            "fitness_over_generations.png",
            "music.txt",
            "fitness_data.csv",
        ]:
            subprocess.call(
                [
                    "mv",
                    os.path.join(".", fileName),
                    os.path.join("./generated", task_name, f"{time_stamp}_{fileName}"),
                ]
            )


if __name__ == "__main__":
    task_list = [
        exeName
        for exeName in os.listdir("./bin")
        if not exeName.startswith(".")
        # and "Standard" not in exeName
        # and "1KG" not in exeName
    ]
    # task_list = ["1KGenerations"]

    run_number = 73
    for i in range(run_number):
        for task_name in task_list:
            print(f"Generating {task_name} - {i+1}th music piece.")
            main(task_name)
    print("Done.")
