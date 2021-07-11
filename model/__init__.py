
class Unit(object):
    inp = None  # layer input
    x = None  # layer output
    u = None  # upsample
    d = None  # down_sample

    def __init__(self, inp=None, x=None, u=None, d=None) -> None:
        super().__init__()
        self.inp, self.x, self.u, self.d = inp, x, u, d
