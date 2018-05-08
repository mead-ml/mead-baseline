rm -rf *.egg*
cp setup_$1.py setup.py
pip install -e .
rm setup.py
