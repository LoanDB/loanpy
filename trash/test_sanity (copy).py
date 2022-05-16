#change path to "data", then change back after function executed or failed
def setpath(method):
    @wraps(method)
    def wrapped(*args, **kwargs):
        owd = os.getcwd() #orginal working directory. Change back to this in the end
        os.chdir(os.path.join(os.path.dirname(__file__), "data"))
        try:
            result = method(*args, **kwargs)
        finally:
            os.chdir(owd)
        return result
    return wrapped
