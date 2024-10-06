
# Valostats, A Valorant Match Predictor
#### UQ Computing Society 2024 People's Choice Award Winner
## Maintainers

**Zain Al-Saffi** - Team Lead, Design, Frontend, Data Analysis and ML

**Varun Singh** - Plots and graphs

**Aman Gupta** - Partial frontend 

**Jasnoor Matharu** - Boosting algorithms

**Lehan Ling** - ML, Frontend and Data processing 

**Abhishek Bhattacharjee** - Pickle


## Background
This project started as a joke, it was inspired by a friend's affinity to gamble Twitch channel points and constantly losing, hence we made a predictor to help with his predictions using machine learning

## Pre requisites
For starters get NPM (Node.js) installed on your local machine, then cd into:
```bash
cd vlresports-1.0.4
```
```bash
npm install
```
```bash
npm start
```

Setup anaconda / miniconda on python 3.8.9 for environment manegement, then install the following:
```bash
conda install scikit-learn
```
```bash
conda install numpy
```
```bash
conda install matplotlib
```
```bash
conda install seaborn
```
```bash
conda install streamlit
```
```bash
conda install kaggle
```
```bash
conda install pandas
```

## Kaggle dataset setup
To use the continuous dataset updater through kaggle API, make sure to do the below:
Go to Kaggle.com and log in.
Navigate to your account settings by clicking on your profile picture in the top-right corner and selecting Account.
Scroll down to the API section and click Create New API Token. This will download a kaggle.json file to your computer.

Then run the following:
```bash
mkdir -p ~/.config/kaggle
mv /path/to/kaggle.json ~/.config/kaggle/kaggle.json
chmod 600 ~/.config/kaggle/kaggle.json  # Secure the file

```
This should setup your Kaggle API, Keep in mind the dataset is updated once a month and likely is the model is displaying 50/50 odds it means there is not enough data about either team, meaning they are likely new teams. 

Credits go to Ryanluong1 on kaggle for the valorant dataset, and the Orloxx23 for the vlresports API.
