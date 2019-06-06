from setuptools import setup


def readme():
    with open('README.md') as f:
        return f.read()


setup(name='pydesim',
      version='0.1.1',
      description='Python Discrete-Event Simulator',
      long_description=readme(),
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
        'Intended Audience :: Science/Research',
      ],
      keywords='simulation, communication systems, modelling, des',
      url='https://github.com/larioandr/pydesim',
      author='Andrey Larionov',
      author_email='larioandr@gmail.com',
      license='MIT',
      packages=['pydesim'],
      scripts=[],
      install_requires=[
          'colorama', 'numpy',
      ],
      include_package_data=True,
      zip_safe=False,
      setup_requires=["pytest-runner"],
      tests_require=["pytest", 'numpy'],
    )
