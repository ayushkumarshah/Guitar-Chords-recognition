# Guitar-Chords-recognition
An application that predicts the chords when melspectrograms of guitar sound is fed into a CNN.

## Setting up the project

### 1. Clone the repo

```console
$ git clone https://github.com/ayushkumarshah/Guitar-Chords-recognition.git
$ cd Guitar-Chords-recognition
```

### 2. Installation

### Option 1: Using Conda - Recommended

- Download and install [miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) or
  [anaconda](https://docs.anaconda.com/anaconda/install/) if you don't have conda installed in your system.

- Create and activate a new environment 'tf' using the following command:

    ```console
    $ conda env create -f environment.yml
    $ conda activate tf
    ```

### Option 2: Using Virtual Environment

- Install and virtualenv using pip and create a virtual environment '.venv'

    ```console
    $ pip install virtualenv
    $ virtualenv .venv
    ```

- Activate the virtual environment '.venv' and install required packages

    ```console
    $ source .venv/bin/activate
    $ pip install -r requirements.txt
    ```

## Running the Chords Classifier App (classifier.py)

It uses the trained model `models/model.json` to predict a recorded guitar chord.

- Execute the python file 'classify.py'

    ```console
    $ python classify.py
    ```

- A window is launched as shown below: 

    ![Home Interface](images/Interface-home.png)

- Click record and play a chord. It records for 3 seconds and saves the output wav file to `recording/recorded.wav`. 

- Click play to listen to the recorded sound. 

- Click classify to view the predicted chord along with the melspectrogram of the recorded chord.

- Click clear to record another chord.

See the demo below:
    <div align = 'center'>
        <a href = 'https://www.youtube.com/watch?v=DOCVIk9Ocys'>
            <img src = 'images/app-demo.gif' alt = 'App demo. Click to go to YouTube!' >
        </a>
    </div>

> Click the above video to to go to YouTube and hear the sound as well.

# Training the model (Optional)

If you want to experiment by training the model yourself with your own data or the data used currently, follow the steps
below:

## 1. Download the Dataset or Use your own dataset

The chords dataset was collected from MONTEFIORE RESEARCH GROUP of University of Liège - Montefiore Institute (Montefiore.ulg.ac.be, 2019). The chords dataset consists of 10 types of chords with 200 audio files of each chord.

Run download_data.sh to download the dataset using:

```console
$ chmod +x download_data.sh
$ ./download_data.sh
```

## 2. Install tensorflow-gpu for faster training (If you have GPU)

**Prerequisites**

- Nvidia GPU (GTX 650 or newer)
- CUDA Toolkit v10.0
- CuDNN 7.6.5

Follow instructions on this
[`site`](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#tensorflow-gpu) to install
cuda and cudnn. However, no need to create a new conda environment as you already have created one.

Then Install tensorflow-gpu. Make sure you are inside the conda environment `tf`

```console
$ conda install -c anaconda tensorflow-gpu==1.13.1
```
