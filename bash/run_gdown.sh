#!/bin/bash

URL_GDRIVE='https://drive.google.com/drive/folders/1B-mvp5r6dogbPz7yzaGgs7HFzydOPajN?usp=share_link'
DOWNLOAD_POINT=${1} # The first argument is the mount point

gdown --folder $URL_GDRIVE --output $DOWNLOAD_POINT --remaining-ok
