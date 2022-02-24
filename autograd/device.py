"""

Authors: Wilhelm Ågren <wagren@kth.se>
Last edited: 24-02-2022
"""



__alldevices__ = [
    'cpu',
    'cuda'
]

def _get_device(tpe):
    devices = {}
    devices['cpu'] = CPU()
    devices['cuda'] = CUDA()

    if tpe not in __alldevices__:
        raise ValueError(
                f'specified device type not recognized, device={tpe}')
    
    return devices[tpe]
    

class Device(object):
    def __init__(self, tpe, **kwargs):
        self._tpe = tpe

    def __str__(self):
        return self._tpe.upper()

    def __repr__(self):
        return f'<autograd.device.{self}>'


class CPU(Device):
    def __init__(self, *args, **kwargs):
        super().__init__('cpu')


class CUDA(Device):
    def __init__(self, *args, **kwargs):
        super().__init__('cuda')

