import hamlib

class IC7300:
    def __init__(self):
        hamlib.rig_set_debug(hamlib.RIG_DEBUG_NONE)
        self.rig = hamlib.Rig(307)
        self.rig.set_conf("rig_pathname", "/dev/ttyUSB0")
        self.rig.open()

    def set_mode(self, mode):
        modes = {
            'USB': hamlib.RIG_MODE_USB,
            'CW': hamlib.RIG_MODE_CW
        }
        self.rig.set_mode(modes[mode], 0)

    def ptt_on(self):
        self.rig.set_ptt(1)

    def ptt_off(self):
        self.rig.set_ptt(0)

    def close(self):
        self.rig.close()
