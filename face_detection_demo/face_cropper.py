from pathlib import Path
from typing import Union

import numpy as np
import tensorflow as tf


def load_detector(cap, face_detector_model_path: Union[Path, str]):
    face_detector = FaceCropper(str(face_detector_model_path),
                                gpu_memory_fraction=0.5,
                                visible_device_list='0'
                                )
    try:
        while True:
            ret, frame = cap.read()
            crops, boxes, scores = face_detector(frame, score_threshold=0.3)
            cv2.imshow('frame_1', crops[0])
            # cv2.imshow('frame_2', crops[1])
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("Releasing")

    cap.release()
    cv2.destroyAllWindows()

    return face_detector


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph


def export_to_pb(sess, outputs, dir_path, export_path):
    # Set the learning phase to Test since the model is already trained.
    #  K.set_learning_phase(0)

    # Create prediction signature to be used by TensorFlow Serving Predict API
    # signature = predict_signature_def(inputs={"images": keras_model.input},
    #                                   outputs={l.name: l for l in keras_model.output})

    frozen_graph = freeze_session(sess, output_names=outputs)

    tf.io.write_graph(frozen_graph,
                      str(dir_path),
                      export_path,
                      as_text=False
                      )


class FaceCropper:
    def __init__(self, model_path, gpu_memory_fraction=0.25, visible_device_list='0'):
        """
        Arguments:
            model_path: a string, path to a pb file.
            gpu_memory_fraction: a float number.
            visible_device_list: a string.
        """
        with tf.gfile.GFile(model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        graph = tf.get_default_graph()  # Graph()
        with graph.as_default():
            tf.import_graph_def(graph_def, name='import')

        self.input_image = graph.get_tensor_by_name('import/image_tensor:0')
        self.output_ops = [
            graph.get_tensor_by_name('import/boxes:0'),
            graph.get_tensor_by_name('import/scores:0'),
            graph.get_tensor_by_name('import/num_boxes:0'),
        ]

        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=gpu_memory_fraction,
            visible_device_list=visible_device_list
        )
        config_proto = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)
        self.sess = tf.Session(graph=graph, config=config_proto)

        boxes, scores, num_boxes = self.output_ops

        # Todo: make it n_faces `tf.Variable`
        n_faces = 1
        num_boxes = num_boxes[0]
        boxes = boxes[0][:num_boxes]
        scores = scores[0][:num_boxes]
        mask = tf.argsort(scores, direction='DESCENDING', stable=False, name=None)

        reduced_boxes = tf.gather(boxes, mask)[:n_faces]
        self.scores = tf.gather(scores, mask)[:n_faces]
        zero_vec = tf.zeros(n_faces, dtype=tf.int32)

        self.crops = tf.image.crop_and_resize(self.input_image,
                                              reduced_boxes,
                                              box_ind=zero_vec,
                                              crop_size=np.array([224, 224]),
                                              # name='target_crops'
                                              )
        self.crops = ((self.crops - tf.reduce_min(self.crops)) /
                      (tf.reduce_max(self.crops) - tf.reduce_min(self.crops)))

        # set names for target tensors
        self.crops = tf.identity(self.crops, name='target_crops')
        self.boxes = tf.identity(reduced_boxes, name='target_boxes')
        self.scores = tf.identity(self.scores, name='target_scores')

    def __call__(self, image, score_threshold=0.5):
        h, w, _ = image.shape
        image = np.expand_dims(image, 0)

        crops, boxes, scores = self.sess.run(
            [self.crops, self.boxes, self.scores],
            feed_dict={self.input_image: image}
        )
        return crops, boxes, scores


if __name__ == '__main__':
    import cv2

    from face_detection_demo import root_folder

    face_detector_model_path = root_folder / 'model' / 'model.pb'
    cap = cv2.VideoCapture(0)
    face_detector = load_detector(cap, face_detector_model_path)

    export_to_pb(face_detector.sess,
                 ['target_crops', 'target_boxes', 'target_scores'],  # [face_detector.crops],
                 root_folder / 'model',
                 'with_crops.pb')
