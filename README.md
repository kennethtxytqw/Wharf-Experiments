# Set up
1. Make sure you have python3 installed.
2. cd into the project directory.
3. Set up a virtual environment if desired (highly recommended).
4. `pip install -r requirements.txt` to install the dependencies.

## VirtualEnv
1. `ipython kernel install --user --name=venv` to enable jupyter to use VirtualEnv.
2. `jupyter nbextension enable --py widgetsnbextension --sys-prefix` to enable `ipywidgets`.

# Usage
1. `./combine.py [segment_size] [overlap_size]` -- to combine the data set into a csv file. Segmenting the data is optional.
2. `./evaluate.py` -- to classify and get accuracy result. It outputs a csv file.