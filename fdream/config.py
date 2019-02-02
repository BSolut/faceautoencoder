base_dir = "/home/richard/data_ocr/faces/"

conf = {
    'base_dir': base_dir,
    'dlib_face_landmark': base_dir+'shape_predictor_5_face_landmarks.dat',
    #'dlib_face_landmark': base_dir+'shape_predictor_68_face_landmarks.dat',

    'data_raw': base_dir+'raw/',
    'data_clean': base_dir+'clean/',
    'data_ignore': base_dir+'ignore/'
}


class Config(object):
    IMAGE_SIZE = 166
    IMAGE_PADDING = 0.4
    PARAM_SIZE = 20

    def __init__(self):
        self._config = conf # set it to conf

    def get_property(self, property_name):
        if property_name not in self._config.keys(): # we don't want KeyError
            return None  # just return None if not found
        return self._config[property_name]

    @property
    def base_dir(self):
        return self.get_property('base_dir')

    @property
    def dlib_face_landmark(self):
        return self.get_property('dlib_face_landmark')

    @property
    def data_raw(self):
        return self.get_property('data_raw')

    @property
    def data_clean(self):
        return self.get_property('data_clean')

    @property
    def data_ignore(self):
        return self.get_property('data_ignore')