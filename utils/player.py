import pygame
import os

class MusicPlayer:
    def __init__(self):
        pygame.mixer.init()
        self.current_song = None

    def load_song(self, song_path):
        self.stop_song()
        pygame.mixer.music.load(song_path)
        self.current_song = song_path

    def play_song(self):
        if self.current_song:
            pygame.mixer.music.play()

    def pause_song(self):
        pygame.mixer.music.pause()

    def unpause_song(self):
        pygame.mixer.music.unpause()

    def stop_song(self):
        pygame.mixer.music.stop()

    def next_song(self, songs, current_index):
        next_index = (current_index + 1) % len(songs)
        self.load_song(songs[next_index])
        self.play_song()
        return next_index

    def prev_song(self, songs, current_index):
        prev_index = (current_index - 1) % len(songs)
        self.load_song(songs[prev_index])
        self.play_song()
        return prev_index
