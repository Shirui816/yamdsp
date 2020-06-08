import atexit
import gc
from collections import OrderedDict


class Ctx:
    systems = OrderedDict()

    @classmethod
    def get_system(cls, num):
        system = cls.systems.get(num, None)
        if system is not None:
            cls.set_active(system)
        return system

    @classmethod
    def destroy(cls, num):
        try:
            system = cls.figs.pop(num)
        except KeyError:
            return
        system.destroy()
        gc.collect(1)

    @classmethod
    def destroy_all(cls):
        import gc
        for system in list(cls.systems.values()):
            system.destroy()
        cls.systems.clear()
        gc.collect(1)

    @classmethod
    def get_num_systems(cls):
        return len(cls.systems)

    @classmethod
    def get_active(cls):
        return next(reversed(cls.systems.values())) if cls.systems else None

    @classmethod
    def set_active(cls, system):
        cls.systems[system.num] = system
        cls.systems.move_to_end(system.num)

    @classmethod
    def has_num(cls, num):
        return num in cls.systems

    @classmethod
    def get_all_systems(cls):
        return list(cls.systems.values())


atexit.register(Ctx.destroy_all)
