import subprocess
import time
import platform
import os
from datetime import datetime
import webbrowser

def wake_computer():
    """Attempt to wake the computer from sleep mode."""
    system = platform.system()
    
    if system == "Windows":
        # On Windows, we can use powercfg
        subprocess.run(["powercfg", "-requestsoverride", "Process", "Python", "System"])
        subprocess.run(["powercfg", "-requestsoverride", "Display", "Python", "System"])
        print("Wake command sent (Windows)")
    
    elif system == "Darwin":  # macOS
        # On macOS, we can use caffeinate to prevent sleep
        subprocess.Popen(["caffeinate", "-u", "-t", "5"])
        print("Wake command sent (macOS)")
    
    elif system == "Linux":
        # On Linux, we can try using rtcwake
        try:
            subprocess.run(["sudo", "rtcwake", "-m", "no", "-s", "2"])
            print("Wake command sent (Linux)")
        except:
            print("Could not send wake command. Make sure rtcwake is installed and you have sudo privileges.")
    
    else:
        print(f"Unsupported operating system: {system}")

def play_alarm(sound_file=None):
    """Play an alarm sound using system commands or browser."""
    system = platform.system()
    
    if sound_file and os.path.exists(sound_file):
        # If a sound file is provided and exists, try to play it with system commands
        print(f"Playing alarm sound: {sound_file}")
        
        if system == "Windows":
            os.startfile(sound_file)  # Windows has a built-in command to open files with default app
        elif system == "Darwin":  # macOS
            subprocess.call(["afplay", sound_file])  # macOS has afplay
        elif system == "Linux":
            # Try different commands that might be available on Linux
            try:
                subprocess.call(["aplay", sound_file])
            except FileNotFoundError:
                try:
                    subprocess.call(["mpg123", sound_file])
                except FileNotFoundError:
                    webbrowser.open(sound_file)  # Last resort: open in browser
        return
    
    # If no sound file is provided or it doesn't exist, use alternative methods
    if system == "Windows":
        # On Windows, use the system alert sound
        import winsound
        for _ in range(5):  # Play 5 times
            winsound.Beep(1000, 1000)  # 1000 Hz for 1 second
            time.sleep(0.5)
    
    elif system == "Darwin":  # macOS
        # Use the system alert on macOS
        for _ in range(5):
            subprocess.call(["afplay", "/System/Library/Sounds/Sosumi.aiff"])
            time.sleep(0.5)
    
    elif system == "Linux":
        # For Linux, try the system beep or open a fallback solution
        try:
            # Try to use system beep
            print('\a\a\a\a\a')
            os.system('beep')  # May require the beep package
        except:
            # Create a simple HTML file with audio and open it in browser
            html_content = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Alarm</title>
                <style>
                    body { background-color: red; text-align: center; padding-top: 20%; }
                    h1 { font-size: 50px; color: white; }
                </style>
                <script>
                    // Create oscillator for alarm sound
                    window.onload = function() {
                        var context = new (window.AudioContext || window.webkitAudioContext)();
                        var oscillator = context.createOscillator();
                        oscillator.type = 'square';
                        oscillator.frequency.setValueAtTime(800, context.currentTime);
                        oscillator.connect(context.destination);
                        oscillator.start();
                        setTimeout(function() { oscillator.stop(); }, 10000);
                    }
                </script>
            </head>
            <body>
                <h1>ALARM!</h1>
            </body>
            </html>
            """
            temp_file = os.path.join(os.path.expanduser("~"), "alarm.html")
            with open(temp_file, "w") as f:
                f.write(html_content)
            webbrowser.open('file://' + temp_file)

def main():
    # You can schedule this to run at a specific time
    scheduled_time = input("Enter alarm time (HH:MM) or press Enter to trigger immediately: ")
    sound_file = input("Enter path to sound file (or press Enter for default sound): ")
    
    if not sound_file:
        sound_file = None
    
    if scheduled_time:
        try:
            hour, minute = map(int, scheduled_time.split(':'))
            now = datetime.now()
            alarm_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            
            # If the time has already passed today, schedule for tomorrow
            if alarm_time < now:
                print("The specified time has passed for today. Setting alarm for tomorrow.")
                alarm_time = alarm_time.replace(day=alarm_time.day + 1)
            
            # Calculate seconds to wait
            wait_seconds = (alarm_time - now).total_seconds()
            
            print(f"Alarm scheduled for {alarm_time.strftime('%H:%M')} "
                  f"({wait_seconds:.0f} seconds from now)")
            time.sleep(wait_seconds)
        except Exception as e:
            print(f"Error parsing time: {e}")
            print("Running alarm immediately")
    
    print("Attempting to wake computer...")
    wake_computer()
    
    print("Playing alarm...")
    play_alarm(sound_file)

if __name__ == "__main__":
    main()