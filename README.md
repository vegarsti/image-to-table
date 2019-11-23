# image-to-table

## Installation

### Install poetry for Python package handling
- Install pyenv (`brew install pyenv`)
- Make sure your global Python is 3, preferably 3.7: `pyenv global 3.7.4`
- Ensure it loads properly in the shell by checking output of `which python`, to fix you might need to add the following snippet to your shell init script (e.g. zshrc)

```
# Pyenv init when creating new shell
eval "$(pyenv init -)"
```

- Install poetry for Python package management
- In this repo, do `poetry install`

## Running

`poetry run python image_with_word_boxes.py` shows the example image (using OpenCV) with bounding boxes around each word.

Service account:
- `export GOOGLE_APPLICATION_CREDENTIALS="service-account.json"`