from abc import ABC



class WorldState:

    def __init__(self) -> None:
        self.dishwasher_exists = True

        ##### replaces direct use of predicate functions for faster runtime ############################################
        self.perceived_objects = []
        """
        objects detected via robot perception
        """

        self.perceived_furniture = []

        self.reachable_objects = []
        """
        objects detected via robot perception, that are reachable from one of the observe positions
        """

        self.misplaced_objects = []
        """
        objects that are at the wrong location, that are misplaced
        """

        self.known_furniture = []
        """
        furniture and walls existing in the world
        """

        self.activated_tasks = []
        """
        all tasks that were triggered
        """

        self.locations = []
        self.trigger_nr = 0
        self.surface_annotation_cache = {}
        self.support_relation_cache = {}
        self.perceived_objects_changed = False

