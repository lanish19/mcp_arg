class FastMCP:
    def __init__(self, name: str):
        self.name = name

    def tool(self, func=None):
        if func is None:
            def deco(f):
                return f
            return deco
        return func

    def resource(self, *args, **kwargs):
        def deco(f):
            return f
        return deco

    def prompt(self, *args, **kwargs):
        def deco(f):
            return f
        return deco


