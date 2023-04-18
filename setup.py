from setuptools import setup, find_packages
import sysconfig

platform = sysconfig.get_platform()
is_m1_mac = platform.startswith("mac") and "universal" in platform or platform == "macosx-11.1-arm64"

if is_m1_mac:
    install_requires = [
        'DynamicFieldPy',
        'tensorflow-macos==2.8.0',
        'matplotlib==3.5.1',
        'numpy==1.22.3',
        'tensorflow-probability==0.15.0',
        'protobuf==3.20.0'
    ]
elif platform == "macosx-11.1-arm64" or platform == "linux-x86_64":
    install_requires = [
        'DynamicFieldPy',
        'tensorflow-macos==2.9.0',
        'matplotlib==3.5.1',
        'numpy==1.22.3',
        'tensorflow-probability==0.15.0',
        'protobuf==3.20.0'
    ]
else:
    install_requires = [
        'DynamicFieldPy',
        'tensorflow==2.8.0',
        'matplotlib==3.5.1',
        'numpy==1.22.3',
        'tensorflow-probability==0.15.0',
        'protobuf==3.20.0'
    ]


setup(name='DynamicFieldFlow',
      version='0.1',
      description='A library for simulating Dynamic Field architectures with TensorFlow',
      url='https://github.com/danielsabinasz/DynamicFieldFlow',
      author='Daniel Sabinasz',
      author_email='daniel@sabinasz.net',
      license='CC-BY-ND 3.0',
      packages=find_packages(include=['dff', 'dff.*']),
      install_requires=install_requires,
      zip_safe=False)
