from camera_movement_estimator.camera_movement_estimator import draw_camera_movement
from trackers.tracker import draw_annotations
from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner
from camera_movement_estimator import CameraMovementEstimator
import cv2


def main():
    # read video
    video_frames = read_video('input_videos/input_video.mp4')

    # initialize tracker
    tracker = Tracker('models/best.pt')

    # cv2.imwrite(f'output_videos/first_frame.jpg', video_frames[0])

    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/track_stubs.pkl')

    # camera movement estimation
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                              read_from_stub=True,
                                                                              stub_path='stubs/camera_movement_stubs.pkl')

    # interpolate ball positions
    # tracks['Ball'] = tracker.interpolate_ball_position(tracks['Ball'])

    # assign player teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['Player'][0])

    for frame_num, player_track in enumerate(tracks['Player']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            tracks['Player'][frame_num][player_id]['team'] = team
            tracks['Player'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # draw output
    # draw object tracks
    output_video_frames = draw_annotations(video_frames, tracks)

    # draw camera movement
    output_video_frames = draw_camera_movement(output_video_frames, camera_movement_per_frame)

    # save video
    save_video(output_video_frames, 'output_videos/output_video.mp4')


if __name__ == '__main__':
    main()
