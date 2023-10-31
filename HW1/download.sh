#!/bin/bash

paragraph_model_google_drive_url="https://drive.google.com/uc?id=1lI6K8u5g9BHbdd7rgVUYFX01G_T2WFWU"
span_model_google_drive_url="https://drive.google.com/uc?id=1WfX3hu8tR_M_qpQByDHr4RqPGuX7kgkL"
datas_google_drive_url="https://drive.google.com/uc?id=1bdPIKrZ6pcPtkyRVM_h-apQNGNZKuiUp"

echo "****************Start Downloading Paragraph Selection Model****************"
gdown $paragraph_model_google_drive_url
echo "****************Finish Downloading Paragraph Selection Model****************"

echo "****************Start Downloading Span Selection Model****************"
gdown $span_model_google_drive_url
echo "****************Finish Downloading Span Selection Model****************"

echo "****************Start Downloading inference utils****************"
gdown $datas_google_drive_url
echo "****************Finish Downloading inference utils****************"
