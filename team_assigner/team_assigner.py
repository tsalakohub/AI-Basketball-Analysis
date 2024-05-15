from sklearn.cluster import KMeans
import numpy as np


def get_clustering_model(image):
    image_2d = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=2, init='k-means++', n_init=1)
    kmeans.fit(image_2d)
    return kmeans


def get_player_color(frame, bbox):
    image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

    jersey = image[int(image.shape[0]/5):int(3*image.shape[0]/5), 
                   int(image.shape[1]/5):int(4*image.shape[1]/5)]

    kmeans = get_clustering_model(jersey)

    labels = kmeans.labels_

    # clustered_image = labels.reshape(jersey.shape[0], jersey.shape[1])

    # corner_clusters = [clustered_image[0, 0],
    #                    clustered_image[0, -1],
    #                    clustered_image[-1, 0],
    #                    clustered_image[-1, -1]]
    # non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
    # player_cluster = 1 - non_player_cluster

    unique_labels, counts = np.unique(labels, return_counts=True)
    most_prevalent_index = np.argmax(counts)
    most_prevalent_label = unique_labels[most_prevalent_index]

    player_color = kmeans.cluster_centers_[most_prevalent_label]

    return player_color


class TeamAssigner:
    def __init__(self):
        self.kmeans = None
        self.team_colors = {}
        self.player_team_dict = {}

    def assign_team_color(self, frame, player_detections):

        player_colors = []

        for _, player_detection in player_detections.items():
            bbox = player_detection['bbox']
            player_color = get_player_color(frame, bbox)
            player_colors.append(player_color)

        kmeans = KMeans(n_clusters=2, init='k-means++', n_init=1)
        kmeans.fit(player_colors)

        self.kmeans = kmeans

        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame, player_bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = get_player_color(frame, player_bbox)

        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
        team_id += 1

        self.player_team_dict[player_id] = team_id

        return team_id
