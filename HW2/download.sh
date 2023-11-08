#!/bin/bash
summarization_model_google_drive_url="https://drive.google.com/uc?id=1tbgzAJYJlGtwWxsU10ENKc0rKHYe5Ki0"
data_google_drive_url="https://drive.google.com/uc?id=1POVwkM30pLoteDJiTiplYsUL2BA_wZcZ"

echo "****************Start Downloading Summarization Model****************"
gdown $summarization_model_google_drive_url
echo "****************Finish Downloading Summarization Model****************"

echo "****************Start Downloading Datas****************"
gdown $data_google_drive_url
echo "****************Finish Downloading Datas****************"
