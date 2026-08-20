from setuptools import setup
from glob import glob

package_name = 'natural_language_processing'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/web', ['resource/stt_visualizer.html']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='petr',
    maintainer_email='petr.vanc@cvut.cz',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "nl_node = natural_language_processing.nl_node:main",
            "stt_node = natural_language_processing.speech_to_text.stt_node:main",
            "stt_visualizer = natural_language_processing.stt_visualizer:main",
            "tts_node = natural_language_processing.text_to_speech.tts_node:main",
        ],
    },
)
