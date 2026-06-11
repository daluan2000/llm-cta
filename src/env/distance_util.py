import math


class DistanceUtil:
    


    def shortest_distance(
            self, location1: tuple[float, float], location2: tuple[float, float]
    ) -> float:
        
        return self.haversine_distance(location1, location2)

    def haversine_distance(
            self, location1: tuple[float, float], location2: tuple[float, float]
    ) -> float:
        
        lon1, lat1 = location1
        lon2, lat2 = location2

        
        lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

        
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))

        
        r = 6371
        return c * r


distance_util = DistanceUtil()
