## Instructions for using "test" Anaconda environment

NOTE: This environment uses the *intel* channel so it may not be the most portable, but if you want to try loading and activating it here are the steps that worked for me:

 - [optional] Copy `test.yml` to your home directory (`%HOMEPATH%` for Windows). In my experience, this matches the default location for Anaconda Terminal, so `<path to test.yml>` below can just be the relative path `test.yml`.
 - Launch the conda terminal:
    - go to the Environments tab of Anaconda Navigator
    - click the green play button
    - select Open Terminal
 - Execute: `conda env create -f <path to test.yml>`
 - Activate the "test" environment by selecting it on the Environments tab of Anaconda Navigator.
