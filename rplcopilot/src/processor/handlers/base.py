class BaseHandler:
    def __init__(self, store):
        self.store = store

    def handle(self, data):
        raise NotImplementedError("Handle method must be implemented by subclass.")
