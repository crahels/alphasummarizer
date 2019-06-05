class PAS:
    def __init__(self):
        self.subjects = []
        self.predicates = []
        self.objects = []
        self.explanations = []
        self.times = []
        self.places = []

    def __init__(self, subjects, predicates, objects, explanations, times, places):
        self.subjects = subjects
        self.predicates = predicates
        self.objects = objects
        self.explanations = explanations
        self.times = times
        self.places = places

    def __str__(self):
        return ("subjects=" + str(self.subjects) + ", predicates=" + str(self.predicates) + 
            ", objects=" + str(self.objects) + ", explanations=" + str(self.explanations) + 
            ", times=" + str(self.times) + ", places=" + str(self.places))
