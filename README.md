Run `pip install -r requirements.txt`
Also run `pip install torch-scatter -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__.split('+')[0])").html`

Now run train.py again (Do above steps only once)
Run train_n.py instead of train.py

To do : 1)Make custom config files for different datasets to reproduce results
2)Implement graph rewiring algorithms
