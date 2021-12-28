import os 
BASE_LINK = "http://158.109.8.102/AuTSL/data/train/train_set_vfbha39.zip."
START_PART = 3
END_PART = 5
for i in range(START_PART, END_PART):
    PART = "0"
    if (i >= 1 and i < 10):
        PART = PART + "0" + str(i)
    else:
        PART = PART + str(i)
    LINK = BASE_LINK + PART
    os.system(f"wget {LINK}")
    # !wget $LINK


# to unzip

"""
sudo apt-get update

sudo apt-get install p7zip-full

7z x train_set_vfbha39.zip.001

password : MdG3z6Eh1t

"""