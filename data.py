import os, sys, argparse
import numpy as np
#import dlib
import cv2
import urllib
from fdream import Config

cfg = Config()

class DataBuilder(object):
    def __init__(self, cfg):
        self.cfg = cfg
        parser = argparse.ArgumentParser(
            description='Data collector',
            usage='''data <command> [<args>]

Command list:
   get        download a list of files and store in data_raw path
   process    crop/align/resize all images in data_raw path
   check      hand cleanup auto processed images
   build      generates train_data from images in data_clean path
   display    display images in train_data
''')
        parser.add_argument('command', help='Subcommand to run')
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, "cmd_"+args.command):
            print 'Unrecognized command'
            parser.print_help()
            exit(1)

        getattr(self, "cmd_" + args.command)()

    def cmd_get(self):
        parser = argparse.ArgumentParser(description='Download a list of files and store in data_raw path')
        parser.add_argument("-s", "--source", required = True, help = "List with links")
        args = parser.parse_args(sys.argv[2:])

        with open(args['source']) as f:
            lines = f.readlines()
        lines = [x.strip() for x in lines]

        line_count = len(lines)
        last_progress = -1


        print("Start downlading {0} files".format(line_count))

        for i, link in enumerate(lines):
            name = os.path.basename(link)
            if not name.endswith('.jpg') and not name.endswith('.png'):
                continue
            if os.path.isfile(cfg.data_raw + name): #allready downloaded
                continue
            urllib.urlretrieve(link, cfg.data_raw + name)
            progress = round((i / float(line_count))*100, 2)
            if progress != last_progress:
                sys.stdout.write('\r')
                sys.stdout.write(str(progress) + "% "+name)
                sys.stdout.flush()

    def cmd_process(self):
        self.face_cascade = cv2.CascadeClassifier(cfg.base_dir+'haarcascade_frontalface_default.xml')
        self.faca_detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(cfg.dlib_face_landmark)
        self.process_all()

    def cmd_check(self):
        self.clean()

    def cmd_build(self):
        parser = argparse.ArgumentParser(description='generates train_data.npy from images in data_clean path')
        parser.add_argument('-noflip', action='store_true', help="Do not add the same image fliped")
        args = vars(parser.parse_args(sys.argv[2:]))

        cfg = self.cfg
        input_files = []
        for name in os.listdir(cfg.data_clean):
            if name.endswith('.jpg') or name.endswith('.png'):
                input_files.append(name)

        num_images = len(input_files)
        num_samples = num_images
        if not args['noflip']:
            num_samples *= 2


        print("Found {0} input files generate {1} samples".format(num_images, num_samples))

        train_data = np.empty((num_samples, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, 3), dtype=np.uint8)
        data_idx = 0

        for idx, name in enumerate(input_files):
            file_name = cfg.data_clean+name
            if not os.path.isfile(file_name):
                continue
            img = cv2.imread(file_name)
            if img is None or len(img.shape) != 3 or img.shape[2] != 3:
                assert(False)
            if img.shape[0] != cfg.IMAGE_SIZE or img.shape[1] != cfg.IMAGE_SIZE:
                assert(False)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            train_data[data_idx] = img
            data_idx += 1

            if not args['noflip']:
                train_data[data_idx] = np.flip(train_data[data_idx -1], axis = 2)
                data_idx += 1

        assert(data_idx == num_samples)
        print("Saving...")
        np.save('train_data.npy', train_data)      

    def cmd_display(self):
        train_data = np.load('train_data.npy')
        idx = 0

        while True:
            file_cnt = len(train_data)
            if idx >= file_cnt:
                idx = 0
            elif idx < 0:
                idx = file_cnt-1
            file_name = train_data[idx]

            sys.stdout.write('\r')
            sys.stdout.write("Index: {0} total count: {1}".format(idx, file_cnt))
            sys.stdout.flush()

            show_img = cv2.cvtColor(train_data[idx], cv2.COLOR_RGB2BGR)
            cv2.imshow('frame', cv2.resize(show_img, (int(show_img.shape[0]*1.5),int(show_img.shape[0]*1.5))))

            key = cv2.waitKey() & 0xff
            if key == 83 or key == 54: #right or keypad 6
                idx += 1
            elif key == 81 or key == 52: #left or keypad 4
                idx -= 1
            elif key == 27: #esc key
                break

    def is_gray(self, img, thumb_size=40, MSE_cutoff=22):
        thumb = cv2.resize(img, (thumb_size, thumb_size))
        SSE = 0
        for y in range(thumb_size):
            for x in range(thumb_size):
                pixel = thumb[y,x]
                mu = sum(pixel)/3
                SSE += sum((pixel[i] - mu)*(pixel[i] - mu) for i in [0,1,2])
        MSE = float(SSE)/(thumb_size*thumb_size)
        return MSE <= MSE_cutoff

    def process(self, file_name, show_debug = True):
        img = cv2.imread(file_name)
        if self.is_gray(img):
            return False

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        dets = self.faca_detector(img_rgb)
        if len(dets) == 0:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            dets = []
            for (x,y,w,h) in faces:
                dets.append( dlib.rectangle(left=x, top=y, right=x+w, bottom=y+h) )

        result = []
        for i, det in enumerate(dets):
            if det.right() - det.left() < self.cfg.IMAGE_SIZE-20:
                continue

            faces = dlib.full_object_detections()
            faces.append(self.shape_predictor(img_rgb, det))

            target = dlib.get_face_chip(img_rgb, faces[0], size=self.cfg.IMAGE_SIZE, padding=self.cfg.IMAGE_PADDING)
            target = cv2.cvtColor(target, cv2.COLOR_RGB2BGR)

            result.append(target)
            if show_debug:
                cv2.imshow('frame', cv2.resize(target, (self.cfg.IMAGE_SIZE*2, self.cfg.IMAGE_SIZE*2)))
                cv2.waitKey(1)

        return result

    def process_all(self):
        last_progress = -1
        all_files = os.listdir(cfg.data_raw)
        for fidx, name in enumerate(all_files):
            progress = round((fidx / float(len(all_files)))*100, 2)
            if progress != last_progress:
                sys.stdout.write('\r')
                sys.stdout.write(str(progress) + "% "+name)
                sys.stdout.flush()
                last_progress = progress

            file_name = cfg.data_raw+name
            if not os.path.isfile(file_name):
                continue
            if os.path.isfile(cfg.data_clean+name):
                continue

            ret = self.process(file_name)
            if not ret:
                continue
            for i, target in enumerate(ret):
                if i == 0:
                    cv2.imwrite(cfg.data_clean+name, target)
                else:
                    cv2.imwrite(cfg.data_clean+str(i)+"_"+name, target)

    def clean(self):
        file_list = []
        for name in os.listdir(cfg.data_clean):
            file_name = cfg.data_clean+name
            if not os.path.isfile(file_name):
                continue
            file_list.append(file_name)

        overlay = cv2.imread('./data/overlay.png')
        idx = 0

        while True:
            file_cnt = len(file_list)
            if idx >= file_cnt:
                idx = 0
            elif idx < 0:
                idx = file_cnt-1
            file_name = file_list[idx]

            sys.stdout.write('\r')
            sys.stdout.write("Index: {0} total count: {1} position: {2} name: {3}".format(idx, file_cnt,
                             str(round(idx/float(file_cnt)*100,2))+"%", os.path.basename(file_name)))
            sys.stdout.flush()

            img = cv2.imread(file_name)
            show_img = cv2.addWeighted(img, 1, overlay, 0.5, 0)
            cv2.imshow('frame', cv2.resize(show_img, (int(show_img.shape[0]*1.5),int(show_img.shape[0]*1.5))))

            key = cv2.waitKey() & 0xff
            if key == 114: # R for remove
                file_list.remove(file_name)
                os.remove(file_name)
            elif key == 83 or key == 54: #right or keypad 6
                idx += 1
            elif key == 81 or key == 52: #left or keypad 4
                idx -= 1
            elif key == 27: #esc key
                break






if __name__ == '__main__':
    DataBuilder(cfg)

