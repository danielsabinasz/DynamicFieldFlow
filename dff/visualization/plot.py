
class Plot:
    def __init__(self, step):
        self._step = step

    def draw(self, value):
        pass

    @property
    def figure(self):
        return self._figure

    @property
    def axes(self):
        return self._axes
