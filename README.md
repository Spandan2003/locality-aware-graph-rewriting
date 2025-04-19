Run `pip install -r requirements.txt`
Also run `pip install torch-scatter -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__.split('+')[0])").html`
Next go to Python\lib\site-packages\graphgps\optimizer\extra_optimizers.py
OR just run and you will get the file where the error happens.
Completely comment out entire file.
Now run train.py again (Do above steps only once)