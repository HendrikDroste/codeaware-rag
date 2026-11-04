from helpers import first as renamed, second, First, Second


def entry():
    label = "ä"; return renamed()


def methods():
    a = First()
    b = Second()
    return a.run(), b.run()


def outer():
    def inner():
        return renamed()
    return inner()


def dynamic(callback):
    return callback()


def ambiguous(flag):
    chosen = renamed if flag else second
    return chosen()


from helpers import First as Parent
import helpers


class Child(Parent, Second):
    pass


class Grandchild(Child):
    pass


class ÄChild(helpers.First, metaclass=type):
    pass


class External(object):
    pass


class Dynamic(unknown()):
    pass


class DynamicAttribute(unknown().First):
    pass


chosen_base = Parent if unknown_flag else Second


class Ambiguous(chosen_base):
    pass
