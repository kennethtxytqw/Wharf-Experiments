# Set up
1. Make sure you have python3 installed.
2. cd into the project directory.
3. Set up a virtual environment if desired (highly recommended, see [VirtualEnv](#virtualenv)).
4. `pip install -r requirements.txt` to install the dependencies.

## VirtualEnv
1. `pip install virtualenv`
2. `virtualenv -p $(which python3) venv`
3. `source venv/bin/activate` activate virtualenv of this project.
4. `pip install -r requirements.txt` to install the dependencies.
5. `ipython kernel install --user --name=venv` to enable jupyter to use VirtualEnv.
6. `jupyter nbextension enable --py widgetsnbextension --sys-prefix` to enable `ipywidgets`.
7. `deactivate` to deactivate virtualenv.

# Usage
1. `jupyter-notebook` to open Jupyter Notebook.
