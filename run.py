import sys

from face_detection_demo.main import main

argv = sys.argv
# Todo: quite raw but allows to send a path to the model weights as the first argument
if len(argv) > 1:
    main(argv[1])
else:
    main()
