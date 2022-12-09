"""
    Class to extract features from audio with MusiCNN and VGG. Based on the musicnn package

    Small differences in the output compared to the original musicnn code come from:
    - audio loading and resampling that uses ffmpeg instead of librosa
    - not converting spectrograms to float16 (keeping float32)

"""

import os

import librosa
import numpy as np
import tensorflow as tf

# disable eager mode for tf.v1 compatibility with tf.v2
tf.compat.v1.disable_eager_execution()

import audio2numpy as a2n
import musicnn
from musicnn import configuration as config
from musicnn import extractor, models
import soxr

config.BATCH_SIZE = 4


def load_audio(
    filename, sr=config.SR, offset=0, duration=None, crash_on_short_duration=True
):
    x, orig_sr = a2n.audio_from_file(filename, offset, duration)
    actual_duration = x.shape[0] / orig_sr
    if duration > actual_duration:
        raise ValueError(f"audio file {filename} is too short for loading {duration}s")
    xr = soxr.resample(x, orig_sr, sr, "HQ")
    return xr.mean(-1)


class FeaturesExtractor:
    def __init__(self, input_length=3, input_overlap=False):

        self.model_list = ["MSD_musicnn", "MTT_musicnn", "MSD_vgg", "MTT_vgg"]

        self.input_overlap = input_overlap

        self.tf_engine = {}
        # with g_mtt.as_default() as g:
        for model in self.model_list:
            # model = 'MTT_musicnn'
            x, extract_vector, is_training, sess = self.get_input_output(
                model, input_length=input_length, input_overlap=input_overlap
            )
            self.tf_engine[model] = {
                "x": x,
                "extract_vector": extract_vector,
                "is_training": is_training,
                "sess": sess,
            }

    def batch_data(self, audio, sr, n_frames, overlap):
        """For an efficient computation, we split the full music spectrograms in patches of length n_frames with overlap.

        INPUT

        - file_name: path to the music file to tag.
        Data format: string.
        Example: './audio/TRWJAZW128F42760DD_test.mp3'

        - n_frames: length (in frames) of the input spectrogram patches.
        Data format: integer.
        Example: 187

        - overlap: ammount of overlap (in frames) of the input spectrogram patches.
        Note: Set it considering n_frames.
        Data format: integer.
        Example: 10

        OUTPUT

        - batch: batched audio representation. It returns spectrograms split in patches of length n_frames with overlap.
        Data format: 3D np.array (batch, time, frequency)

        - audio_rep: raw audio representation (spectrogram).
        Data format: 2D np.array (time, frequency)
        """

        # compute the log-mel spectrogram with librosa
        audio_rep = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            hop_length=config.FFT_HOP,
            n_fft=config.FFT_SIZE,
            n_mels=config.N_MELS,
        ).T
        # audio_rep = audio_rep.astype(np.float16)
        audio_rep = np.log10(10000 * audio_rep + 1)

        # batch it for an efficient computing
        first = True
        last_frame = audio_rep.shape[0] - n_frames + 1
        # +1 is to include the last frame that range would not include
        for time_stamp in range(0, last_frame, overlap):
            patch = np.expand_dims(
                audio_rep[time_stamp : time_stamp + n_frames, :], axis=0
            )
            if first:
                batch = patch
                first = False
            else:
                batch = np.concatenate((batch, patch), axis=0)

        return batch, audio_rep

    def get_input_output(self, model, input_length=3, input_overlap=False):

        # select model
        if "MTT" in model:
            labels = config.MTT_LABELS
        elif "MSD" in model:
            labels = config.MSD_LABELS
        num_classes = len(labels)

        if "vgg" in model and input_length != 3:
            raise ValueError(
                "Set input_length=3, the VGG models cannot handle different input lengths."
            )

        # convert seconds to frames
        n_frames = (
            librosa.time_to_frames(
                input_length,
                sr=config.SR,
                n_fft=config.FFT_SIZE,
                hop_length=config.FFT_HOP,
            )
            + 1
        )

        # tensorflow: define the model
        g = tf.Graph()
        with g.as_default() as g:
            with tf.name_scope("model"):
                x = tf.compat.v1.placeholder(
                    tf.float32, [None, n_frames, config.N_MELS]
                )
                is_training = tf.compat.v1.placeholder(tf.bool)
                if "vgg" in model:
                    y, pool1, pool2, pool3, pool4, pool5 = models.define_model(
                        x, is_training, model, num_classes
                    )
                else:
                    (
                        y,
                        timbral,
                        temporal,
                        cnn1,
                        cnn2,
                        cnn3,
                        mean_pool,
                        max_pool,
                        penultimate,
                    ) = models.define_model(x, is_training, model, num_classes)

            sess = tf.compat.v1.Session()
            sess.run(tf.compat.v1.global_variables_initializer())
            saver = tf.compat.v1.train.Saver()
            saver.restore(sess, os.path.dirname(musicnn.__file__) + "/" + model + "/")

        if "vgg" in model:
            extract_vector = [pool1, pool2, pool3, pool4, pool5]
        else:
            extract_vector = [mean_pool, max_pool, penultimate]

        return x, extract_vector, is_training, sess

    def extract_features(self, audio_data, model="MTT_musicnn"):
        """Extract the taggram (the temporal evolution of tags) and features (intermediate representations of the model) of the music-clip in file_name with the selected model.

        INPUT

        - input_data: numpy array. Spectrogram formatted by batch_data

        - model: select a music audio tagging model.
        Data format: string.
        Options: 'MTT_musicnn', 'MTT_vgg', 'MSD_musicnn', 'MSD_musicnn_big' or 'MSD_vgg'.
        MTT models are trained with the MagnaTagATune dataset.
        MSD models are trained with the Million Song Dataset.
        To know more about these models, check our musicnn / vgg examples, and the FAQs.
        Important! 'MSD_musicnn_big' is only available if you install from source: python setup.py install.

        - input_length: length (in seconds) of the input spectrogram patches. Set it small for real-time applications.
        Note: This is the length of the data that is going to be fed to the model. In other words, this parameter defines the temporal resolution of the taggram.
        Recommended value: 3, because the models were trained with 3 second inputs.
        Observation: the vgg models do not allow for different input lengths. For this reason, the vgg models' input_length needs to be set to 3. However, musicnn models allow for different input lengths: see this jupyter notebook.
        Data format: floating point number.
        Example: 3.1

        - input_overlap: ammount of overlap (in seconds) of the input spectrogram patches.
        Note: Set it considering the input_length.
        Data format: floating point number.
        Example: 1.0


        OUTPUT

        - taggram: expresses the temporal evolution of the tags likelihood.
        Data format: 2D np.ndarray (time, tags).
        Example: see our basic / advanced examples.

        - tags: list of tags corresponding to the tag-indices of the taggram.
        Data format: list.
        Example: see our FAQs page for the complete tags list.

        - features: if extract_features = True, it outputs a dictionary containing the activations of the different layers the selected model has.
        Data format: dictionary.
        Keys (musicnn models): ['timbral', 'temporal', 'cnn1', 'cnn2', 'cnn3', 'mean_pool', 'max_pool', 'penultimate']
        Keys (vgg models): ['pool1', 'pool2', 'pool3', 'pool4', 'pool5']
        Example: see our musicnn and vgg examples.

        """

        x = self.tf_engine[model]["x"]
        extract_vector = self.tf_engine[model]["extract_vector"]
        is_training = self.tf_engine[model]["is_training"]
        sess = self.tf_engine[model]["sess"]

        n_frames = x.shape[1]

        if not self.input_overlap:
            overlap = n_frames
        else:
            overlap = librosa.time_to_frames(
                self.input_overlap,
                sr=config.SR,
                n_fft=config.FFT_SIZE,
                hop_length=config.FFT_HOP,
            )

        # batching data
        print("Computing spectrogram (w/ librosa) and tags (w/ tensorflow)..", end=" ")
        batch, _ = self.batch_data(audio_data, config.SR, n_frames, overlap)

        # tensorflow: extract features and tags
        # ..first batch!

        tf_out = sess.run(
            extract_vector,
            feed_dict={x: batch[: config.BATCH_SIZE], is_training: False},
        )

        if "vgg" in model:
            features = dict()
            for layer in range(5):
                features[f"pool{layer+1}"] = tf_out[layer][:, :, 0, :]

        else:
            mean_pool_, max_pool_, penultimate_ = tf_out
            features = dict()
            features["mean_pool"] = mean_pool_
            features["max_pool"] = max_pool_
            features["penultimate"] = penultimate_

        # ..rest of the batches!
        for id_pointer in range(config.BATCH_SIZE, batch.shape[0], config.BATCH_SIZE):

            tf_out = sess.run(
                extract_vector,
                feed_dict={
                    x: batch[id_pointer : id_pointer + config.BATCH_SIZE],
                    is_training: False,
                },
            )

            if "vgg" in model:
                for layer in range(5):
                    features[f"pool{layer+1}"] = np.concatenate(
                        (features[f"pool{layer+1}"], tf_out[layer][:, :, 0, :]), axis=0
                    )
            else:
                mean_pool_, max_pool_, penultimate_ = tf_out
                features["mean_pool"] = np.concatenate(
                    (features["mean_pool"], mean_pool_), axis=0
                )
                features["max_pool"] = np.concatenate(
                    (features["max_pool"], max_pool_), axis=0
                )
                features["penultimate"] = np.concatenate(
                    (features["penultimate"], penultimate_), axis=0
                )

        # sess.close()
        print("done!")

        return features
