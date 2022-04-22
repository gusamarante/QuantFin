import os
import getpass
from pathlib import Path

if os.environ.get('OS') == 'Windows_NT':
    file_path = Path(r'C:\Users\gamarante\Dropbox\Personal Portfolio')
else:  # Assume Mac
    username = getpass.getuser()
    file_path = Path(f'/Users/{username}/Dropbox/Personal Portfolio')

DROPBOX = file_path
