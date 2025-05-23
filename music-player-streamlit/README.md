# Music Player App

This is a simple music player web application built using Streamlit and Python. It allows users to play, pause, stop, and switch between MP3 songs from a predefined playlist. Below is a detailed guide on how to use and understand the structure of this application.

## How to Use

### Setup and Requirements
- Make sure you have Python installed (preferably Python 3.x).
- Install required Python packages using `pip install -r requirements.txt`.

### Running the App
- Clone this repository to your local machine.
- Navigate to the project directory in your terminal.
- Run the Streamlit app with `streamlit run app.py`.
- The app will open in your default web browser.

### Functionality
- **Song Selection**: The app automatically loads MP3 files from the `assets/songs/` directory and displays them in a playlist format.
- **Playback Controls**: Users can control playback using buttons for play, pause, stop, previous song, and next song.
- **Cover Art Display**: The app displays cover images associated with each song in the playlist.
- **Current Song Display**: Shows the title of the currently playing song.

## Files and Directories

- **app.py**: Contains the main Streamlit application code.
- **utils/player.py**: Provides the MusicPlayer class used for controlling song playback.
- **assets/**:
  - **songs/**: Directory where MP3 files are stored.
  - **images/**: Directory where cover images for each song are stored.

## Dependencies

- **Streamlit**: Used for creating the web application interface.
- **mutagen**: Library for reading MP3 metadata.

## Customization

- **Adding Songs**: Simply add MP3 files to the `assets/songs/` directory. Ensure the cover image filenames start with 'cover_image' and are stored in `assets/images/`.
- **Changing UI**: Modify the Streamlit components (e.g., buttons, layout) in `app.py` to customize the user interface.
- **Enhancing Features**: Extend the MusicPlayer class in `utils/player.py` to add more advanced features like volume control, shuffle mode, etc.

## Notes

- This app uses Streamlit's session state to maintain the current song index across user interactions.
- Cover images are sorted and displayed alongside song titles for a more engaging user experience.

## Contributions

Feel free to fork this repository, make improvements, and create pull requests. Contributions are welcome, whether it's fixing bugs, adding features, or enhancing the UI/UX.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Contact

For any questions or feedback, please contact [your email or GitHub handle].
