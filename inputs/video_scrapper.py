from pytubefix.cli import on_progress
from pytubefix import YouTube
import os

url = "https://www.youtube.com/watch?v=2bHsPtENhWc"
RES = '1080p'

yt = YouTube(url, on_progress_callback = on_progress)

for idx,i in enumerate(yt.streams):
   if i.resolution ==RES:
      print(idx)
      print(i.resolution)
      break
   
output_folder = 'la_liga_mateches/Barcelona/'

yt.streams[idx].download(output_path=output_folder)
