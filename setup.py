from distutils.core import setup

setup(
    name='mne_addon',
    version='0.1dev',
    packages=['mne_addon'],
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    long_description=open('README.txt').read(),
    include_package_data=True,
)
