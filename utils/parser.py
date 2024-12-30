import argparse


def parse_args(callback=None):
    """
    Parse command line arguments.

    Args:
        callback: callback function for args of a specific task.

    Returns:
        args: arguments
    """
    args = None
    parser = argparse.ArgumentParser()

    if callback is not None:
        parser = callback(parser)
        args = parser.parse_args()
        print("parse command-line arguments \n")

    return args
