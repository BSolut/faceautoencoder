*A pretrained autoencoder for femalefaces*
[<img src="https://i.imgur.com/BtbrwwT.jpg" />]()
## Run
 Install dependencies [dlib], [cv2], [pygame], [matplotlib], [keras]:
```bash
sudo pip install dlib opencv pygame matplotlib keras
```
```bash
git clone git@github.com:BSolut/faceautoencoder.git
```
Execute the editor
```bash
python editor.py
```
## Train your own dataset
#### Friendly note
If you still want to sleep peacefully at night, make sure that there are only two eyes in one picture in your training data.
[<img src="https://i.imgur.com/0vbzwPY.gif" />]()
#### Building training data
* Build prepare working dir:
  ```bash
  mdir ~/working_dir
  cd ~/working_dir
  mkdir clean
  mkdir raw
  mkdir ignore
  wget https://github.com/davisking/dlib-models/raw/master/shape_predictor_5_face_landmarks.dat.bz2
  bzip2 -d shape_predictor_5_face_landmarks.dat.bz2
  wget https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml
  ```
* Generate a textfile with links for training images. Acceptable sources are [pinterest.com](https://pinterest.com), [famousbirthdays.com](https://www.famousbirthdays.com/) or any source of your linking
 * Download images into a directory
   ```bash
   python data.py get --source [links.txt]
   ```
  * Auto process images (croping/face aligment)
    ```bash
    python data.py process
    ```
  * Remove any outliers (e.g. not a face, black and white images)
    ```bash
    python data.py check
    ```
    Once started, you can use:
    r - removes that image
    Left/Right arrow key - move inside the dataset
    ESC - exits
 * Build train_data.npy
   ```bash
   python data.py build
   ```
#### Execute training
```bash
python train.py
```
Once you are happy with the results, build stats
```bash
python stats.py
```