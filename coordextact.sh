#!/bin/bash

source venv_hmr/bin/activate
for class in data/*; do
    echo "$(basename " $class ")"
    echo "python2 demo.py -m demo --img_path $class"
    python2 -m demo --img_path $class
done
deactivate
echo "python lstm_classifier.py"
python lstm_classifier.py
echo "python pca.py"
python pca.py

