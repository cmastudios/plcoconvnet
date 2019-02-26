import logging
import glob
import re

import numpy as np
import skimage.io
import skimage.util
import cv2
import sklearn.model_selection
import sklearn.utils

logger = logging.getLogger(__name__)


class Dataset(object):
    def __init__(self, name, im_size, classes, max_memory=10**9, test_size=0.25):
        self.name = name
        self.im_size = im_size
        self.classes = classes
        self.class_map = {name:index for index, name in enumerate(self.classes)}
        self.i = 0
        self.test_size = test_size
        num_feat = max_memory // np.prod(im_size) // 8
        self.features = np.zeros((num_feat, np.prod(im_size)), dtype='float64')
        self.labels = np.zeros((num_feat, len(classes)), dtype='float64')
        logger.info('Initialized {} dataset with {} features.'.format(name, num_feat))

    def add_feature(self, feature, label):
        if self.i >= self.features.shape[0]:
            raise RuntimeError('out of space')
            
        if self.features[self.i].any():
            # there's already something there
            bb = self.first_empty_space(self.i)
            if bb is not None:
                self.i = bb
                logger.warning('Updating i to next bad block {}.'.format(self.i))
            else:
                raise RuntimeError('out of space')

        if type(feature) == str:
            feature = skimage.io.imread(feature)

        if len(feature.shape) == 3:
            # color to grayscale
            # feature = skimage.color.rgb2gray(feature)
            feature = cv2.cvtColor(feature, cv2.COLOR_BGR2GRAY)

        if len(feature.shape) == 2:
            if feature.shape != self.im_size:
                # uniform size
                # feature = scipy.misc.imresize(feature, self.im_size, 'bilinear')
                feature = cv2.resize(feature, self.im_size)
            feature = feature.flatten()

        if feature.dtype != 'float64' or feature.max() > 1:
            feature = skimage.util.img_as_float(feature)
           
        if not feature.any():
            logger.warning("Not inserting feature {} as it is all black".format(self.i))
            return

        if not isinstance(label, float):
            label = self.class_map[label]

        self.features[self.i] = feature
        self.labels[self.i][label] = 1.

        self.i += 1
        if self.i % 100 == 0:
            logger.info("Added feature {}.".format(self.i))

    def save(self):
        filename = '{name}_{x}x{y}_{block}.npz'.format(name=self.name, x=self.im_size[0],
                                                       y=self.im_size[1], block=0)
        total = self.first_empty_space(self.i)
        to_delete = self.features.shape[0] - total
        if to_delete > 5:
            logger.info("Clearing {} unused spaces".format(to_delete))
        self.features = self.features[:total, :]
        self.labels = self.features[:total, :]

        np.savez_compressed(filename, features=self.features, labels=self.labels,
                            classes=self.classes, test_size=self.test_size)

    def first_empty_space(self, start=0):
        for i in range(start, self.features.shape[0]):
            if not self.features[i].any():
                return i
        return None

    @classmethod
    def load(cls, name):
        self = cls.__new__(cls)
        file = glob.glob(name + "*.npz")[0]
        result = re.search(r"(\S+)_(\d+)x(\d+)_(\d+)\.npz", file)
        x = int(result.group(2))
        y = int(result.group(3))

        self.name = name
        self.im_size = (x, y)
        self.i = 0

        archive = np.load(file)

        self.classes = archive['classes']
        self.features = archive['features']
        self.labels = archive['labels']
        self.test_size = 0.25
        try:
            self.test_size = archive['test_size']
        except KeyError as e:
            pass
            
        logger.info('Loaded {} dataset with {} features.'.format(self.name, self.features.shape[0]))
        return self

    def batch(self):
        return Batch(self)


class Batch(object):
    def __init__(self, dataset):
        total = dataset.first_empty_space()
        if total is None:
			total = dataset.features.shape[0]
        self.feature_i = 0
        # Split the data into train and validation sections.
        if dataset.test_size == 1:
            self.features_train, self.features_test, self.labels_train, self.labels_test = \
                (dataset.features[:0], dataset.features[:total], dataset.labels[:0], dataset.labels[:total])
        elif dataset.test_size == 0:
            self.features_train, self.features_test, self.labels_train, self.labels_test = \
                (dataset.features[:total], dataset.features[:0], dataset.labels[:total], dataset.labels[:0])
        else:
            self.features_train, self.features_test, self.labels_train, self.labels_test = \
                sklearn.model_selection.train_test_split(dataset.features[:total], dataset.labels[:total],
                                                         test_size=float(dataset.test_size), random_state=0)
        logger.info('Using {trainft} features for training and {valft} features for testing.'.format(
            trainft=self.features_train.shape[0], valft=self.features_test.shape[0]))

    def next_batch(self, count):
        """Advance to the next block of data.

        :param count: Amount of images requested.
        :return: Tuple with features as first element and labels as second element.
        """
        a = self.features_train[self.feature_i:self.feature_i + count]
        b = self.labels_train[self.feature_i:self.feature_i + count]
        if a.shape[0] == 0:
            self.feature_i = 0
            self.features_train, self.labels_train = sklearn.utils.shuffle(self.features_train, self.labels_train)
            return self.next_batch(count)

        self.feature_i += a.shape[0]
        return (a, b)

    @property
    def test(self):
        """The validation dataset, selected as a random 25% of the total."""
        return {'images': self.features_test, 'labels': self.labels_test}

