from Basilisk.utilities import simIncludeGravBody

class GravBodyInterface(object):
    grav_body = None
    grav_file = None
    grav_degree = None
    grav_factory = None

    def __init__(self, planet, grav_file, degree):
        self.planet = planet
        self.grav_factory = simIncludeGravBody.gravBodyFactory()
        self.grav_body= self.grav_factory.createBodies([self.planet])#, 'sun', 'moon', "jupiter barycenter"])
        self.grav_file = grav_file
        self.grav_degree = degree
        return

    def configure(self):
        self.grav_factory.spiceObject.zeroBase =  self.planet.capitalize() #TODO: Check that this consistently works
        self.grav_body[self.planet].isCentralBody = True
        self.grav_body[self.planet].useSphericalHarmParams = True
        return
    
