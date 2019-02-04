import os, pygame, time
import numpy as np
from fdream import Config
from fdream.autoencoder import AutoEncoder

#User constants
background_color = (210, 210, 210)
slider_color = (20, 20, 20)

num_params = 80
slider_w = 15
slider_h = 100
slider_px = 5
slider_py = 10
slider_cols = 20
input_w = 166
input_h = 166
image_scale = 3
image_padding = 10


#Derived constants
slider_w = slider_w + slider_px*2
slider_h = slider_h + slider_py*2
drawing_x = image_padding
drawing_y = image_padding
drawing_w = input_w * image_scale
drawing_h = input_h * image_scale
sliders_w = slider_w * slider_cols
slider_rows = (num_params - 1) / slider_cols + 1
sliders_x = drawing_x + drawing_w + image_padding
sliders_y = image_padding
sliders_h = slider_h * slider_rows
window_w = drawing_w + image_padding*3 + sliders_w
window_h = drawing_h + image_padding*2

class FaceEditor(object):
    def __init__(self, cfg, basedir):
        self.is_running = True
        self.needs_update = True
        self.cfg = cfg
        self.cur_params = np.zeros((num_params,), dtype=np.float32)
        self.cur_face = np.zeros((input_w, input_h, 3), dtype=np.uint8)

        ae = AutoEncoder()
        ae.load(basedir)
        self.model = ae.decoder

        self.means = np.load(basedir+'means.npy')
        self.evals = np.load(basedir+'evals.npy')
        self.evecs = np.load(basedir+'evecs.npy')

        sort_inds = np.argsort(-self.evals)
        self.evals = self.evals[sort_inds]
        self.evecs = self.evecs[:,sort_inds]

        self.mouse_pressed = False
        self.cur_slider_ix = 0

        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((window_w, window_h))
        self.face_surface_mini = pygame.Surface((input_w, input_h))
        self.face_surface = self.screen.subsurface((drawing_x, drawing_y, drawing_w, drawing_h))
        pygame.display.set_caption('Face Editor')

    def process_mouse_click(self, mouse_pos):
        x = (mouse_pos[0] - sliders_x)
        y = (mouse_pos[1] - sliders_y)

        if x >= 0 and y >= 0 and x < sliders_w and y < sliders_h:
            slider_ix_w = int(x / slider_w)
            slider_ix_h = int(y / slider_h)

            self.cur_slider_ix = slider_ix_h * slider_cols + slider_ix_w
            self.mouse_pressed = True

    def process_mouse_move(self, mouse_pos):
        y = (mouse_pos[1] - sliders_y)

        if y >= 0 and y < sliders_h:
            slider_row_ix = int(self.cur_slider_ix / slider_cols)
            slider_val = y - slider_row_ix * slider_h

            slider_val = min(max(slider_val, slider_py), slider_h - slider_py) - slider_py
            val = (float(slider_val) / (slider_h - slider_py*2) - 0.5) * 6.0
            self.cur_params[self.cur_slider_ix] = val

            self.needs_update = True


    def process_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.is_running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == 27:
                    self.is_running = False
                elif event.key == pygame.K_r:
                    self.cur_params = np.clip(np.random.normal(0.0, 1.0, (num_params,)), -3.0, 3.0)
                    self.needs_update = True
                elif event.key == pygame.K_c:
                    self.cur_params = np.zeros((num_params,), dtype=np.float32)
                    self.needs_update = True
                elif event.key == pygame.K_s:
                    i = 0
                    while os.path.isfile("save_"+str(i)+".jpg"):
                        i += 1
                    pygame.image.save(self.face_surface_mini,"save_"+str(i)+".jpg")

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if pygame.mouse.get_pressed()[0]:
                    prev_mouse_pos = pygame.mouse.get_pos()
                    self.process_mouse_click(prev_mouse_pos)
                    self.process_mouse_move(prev_mouse_pos)
                elif pygame.mouse.get_pressed()[2]:
                    self.cur_params = np.zeros((num_params,), dtype=np.float32)
            elif event.type == pygame.MOUSEBUTTONUP:
                self.mouse_pressed = False
                self.needs_update = True
            elif event.type == pygame.MOUSEMOTION and self.mouse_pressed:
                self.process_mouse_move(pygame.mouse.get_pos())

    def draw_sliders(self):
        screen = self.screen
        for i in range(num_params):
            row = int(i / slider_cols)
            col = i % slider_cols
            x = sliders_x + col * slider_w
            y = sliders_y + row * slider_h

            cx = int(x + slider_w / 2)
            cy_1 = y + slider_py
            cy_2 = y + slider_h - slider_py
            pygame.draw.line(screen, slider_color, (cx, cy_1), (cx, cy_2))

            py = y + int((self.cur_params[i] / 6.0 + 0.5) * (slider_h - slider_py*2)) + slider_py
            pygame.draw.circle(screen, slider_color, (cx, py), int(slider_w/2 - slider_px))

            cx_1 = x + slider_px
            cx_2 = x + slider_w - slider_px
            for j in range(7):
                ly = y + slider_h/2 + (j-3)*(slider_h/7)
                pygame.draw.line(screen, slider_color, (cx_1, ly), (cx_2, ly))

    def draw_face(self):
        pygame.surfarray.blit_array(self.face_surface_mini, np.transpose(self.cur_face, (1,0,2)))
        pygame.transform.scale(self.face_surface_mini, (drawing_w, drawing_h), self.face_surface)
        pygame.draw.rect(self.screen, (0,0,0), (drawing_x, drawing_y, drawing_w, drawing_h), 1)


    def update(self):
        if not self.needs_update:
            return

        x = self.means + np.dot(self.cur_params * self.evals, self.evecs)
        x = np.expand_dims(x, axis=0)

        ret = self.model.predict([x])
        self.cur_face = (ret[0]*255.0).astype(np.uint8)

        self.needs_update = False


    def run(self):
        while self.is_running:
            #start = time.time()

            self.process_events()
            if not self.is_running:
                break

            self.update()

            self.screen.fill(background_color)
            self.draw_sliders()
            self.draw_face()

            pygame.display.flip()
            pygame.time.wait(1)
            #pygame.time.wait(max(1./25 - (time.time() - start), 0))



if __name__ == '__main__':
    fe = FaceEditor(Config(), "./weights/")
    fe.run()
