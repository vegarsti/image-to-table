class InvalidData(Exception):
    def __str__(self) -> str:
        return "Request data needs to be a png image"
