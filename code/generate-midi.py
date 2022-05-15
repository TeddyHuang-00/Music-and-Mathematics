import os
import midiutil.MidiFile as midi

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
