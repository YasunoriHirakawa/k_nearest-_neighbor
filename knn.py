from pprint import pprint
import math
import numpy as np


class KNN:

    def __init__(self, k, data):

        self.k = k
        self.data = data
        self.classes = self._extract_classes()

    def _extract_classes(self):

        classes = [d["class"] for d in self.data]

        return list(set(classes))

    def _calc_dist(self, a, b):

        a = np.array(a)
        b = np.array(b)

        return np.linalg.norm(a-b)

    def run(self, sample):

        for d in self.data:
            d["dist"] = self._calc_dist(d["features"], sample)
        sorted_data = sorted(self.data, key=lambda x:x["dist"])

        print("Calculated dists...")
        pprint(sorted_data)
        print()

        container = [{"class": cls, "count": 0} for cls in self.classes]
        for i, d in enumerate(sorted_data):
            if i == self.k:
                break
            for c in container:
                c["count"] += 1 if d["class"] == c["class"] else 0
        sorted_container = sorted(container, key=lambda x:x["count"], reverse=True)

        print("Count of classes in selected data...")
        pprint(sorted_container)
        print()

        return sorted_container[0]["class"]


def main():

    data = [
        {"features": [8, 9], "class": "A"},
        {"features": [8, 4], "class": "A"},
        {"features": [4, 4], "class": "B"},
        {"features": [2, 4], "class": "B"},
        {"features": [7, 7], "class": "A"}
    ]
    sample = [3, 7]

    knn = KNN(3, data)
    cls = knn.run(sample)
    print(f"Predicted Class...\n{cls}")


if __name__ == "__main__":
    main()
