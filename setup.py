from distutils.core import setup


with open('requirements.txt') as f:
    # Todo: read recipe at https://www.reddit.com/r/Python/comments/3uzl2a/setuppy_requirementstxt_or_a_combination/
    requirements = f.read().splitlines()


setup(
    name='face_detection_demo',
    version='0.0.1',
    url='jet.msk.su',
    license='Apache 2',
    author='Mikhail Stepanov',
    author_email='ml.stepanov@jet.msk.su',
    description='face detection demo for CSC',
    include_package_data=True,
    python_requires='>=3.6.5',
    install_requires=requirements,
)
