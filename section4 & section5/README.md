

## Dependencies

### Python
You will require a python3 installation with some specific packages. To install the necessary packages you can run

<pre translate="no" dir="ltr" is-upgraded="">
python -m venv .venv && source .venv/bin/activate
pip3 install -r requirements.txt
</pre>

### Jupyter notebook 
All code is provided via jupyter notebooks via jupyter notebooks. 
To open these files you require the web application *jupyter*, which you can install via 

<pre translate="no" dir="ltr" is-upgraded="">
pip3 install jupyterlab
</pre>

Then install a jupyter kernel for the virtual python environment you installed the packages for via

<pre translate="no" dir="ltr" is-upgraded="">
pip3 install ipykernel
python -m ipykernel install --user --name=venv
</pre>

To then start jupyter just execute

<pre translate="no" dir="ltr" is-upgraded="">
jupyter lab
</pre>

This opens the jupyter web application in your standard browser. Now you can navigate to the corresponding directory and open the respective file.

### Cryptographic Estimators

Concrete estimates were obtained using the cryptographic estimators, which are already installed via the requirements.txt. 
However, we used the most up-to-date version of the estimators which can be installed as follows (otherwise it might be that slighty different numbers are observed)
1. Clone the github repository https://github.com/Crypto-TII/CryptographicEstimators\
2. Switch to branch ```develop``` via 
<pre translate="no" dir="ltr" is-upgraded="">
git checkout develop
</pre>
3. Install the package via
<pre translate="no" dir="ltr" is-upgraded="">
pip3 install .
</pre>