import os
import uuid
from abc import ABC, abstractmethod


class Plot(ABC):
    def __init__(self, fontsize=15):
        self.fontsize = fontsize

    @classmethod
    def get_filepath(cls, filepath=None, filename=None, random=False):
        """
        Build the filepath for saving
        """
        base_path = cls.get_fig_dir()
        if filepath is None:
            if filename is not None:
                filepath = os.path.join(base_path, filename)
                try:
                    os.makedirs(os.path.dirname(filepath))
                except FileExistsError:
                    pass
            elif random:
                gid = uuid.uuid4()
                filepath = os.path.join(base_path, "graph_%s.pdf" % gid)
        return filepath

    @classmethod
    def get_fig_dir(cls, suffix=None):
        """
        Return the base figure path
        """
        base_path = os.path.dirname(os.path.realpath(__file__))
        base_path = os.path.join(base_path, "../../fig/")
        assert os.path.exists(base_path)
        if suffix is not None:
            base_path = os.path.join(base_path, suffix)
            try:
                os.makedirs(os.path.dirname(base_path))
            except FileExistsError:
                pass
        return base_path
