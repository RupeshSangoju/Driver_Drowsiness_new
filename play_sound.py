import vlc
import time

def play_alarm():
    # Initialize VLC player and load the alarm sound
    player = vlc.MediaPlayer("alert-sound.mp3")  # Replace with your alarm file path
    player.play()
    return player

def stop_alarm(player):
    # Stop the alarm
    player.stop()

# Simulating the detection of drowsiness
is_drowsy = True

if is_drowsy:
    print("Drowsiness detected! Playing alarm...")
    player = play_alarm()
    time.sleep(5)  # Let the alarm sound for 5 seconds
    stop_alarm(player)
else:
    print("No drowsiness detected.")
