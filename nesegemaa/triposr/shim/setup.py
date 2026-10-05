from setuptools import setup

setup(
    name="torchmcubes-cpu-shim",
    version="0.1.0",
    description="CPU marching-cubes exposing the tatsy/torchmcubes call convention (via PyMCubes)",
    packages=["torchmcubes"],
    package_dir={"torchmcubes": "torchmcubes"},
    install_requires=["torch", "mcubes", "numpy"],
    python_requires=">=3.8",
)
