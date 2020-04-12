conda create -y -n tf python=3.6
eval "$(conda shell.zsh hook)"
conda activate tf
# source activate tf
conda install -y pandas==1.0.3
conda install -y -c conda-forge librosa=0.6.1
pip install pillow==7.1.1
conda install -y -c anaconda tk==8.6.8
conda install -y -c anaconda pyaudio==0.2.11
pip install tensorflow==1.8.0 keras==2.2.4
pip install imbalanced-learn==0.6.2
pip install streamlit==0.57.3
conda install -y -c conda-forge jupyter notebook
