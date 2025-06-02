from setuptools import setup

setup(
    name="rpl",
    version="0.1.0",
    py_modules=["rpl"],
    install_requires=["typer[all]"],
    entry_points={
        "console_scripts": [
            "rpl = rpl:app",
        ],
    },
)
