def first():
    return "first"


def second():
    return "second"


class First:
    def run(self):
        return first()


class Second:
    def run(self):
        return second()


def unused():
    return None


class Child:
    pass
