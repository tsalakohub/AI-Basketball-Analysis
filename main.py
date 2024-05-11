from utils import read_video, save_video
from trackers import Tracker

def main():
    # read video
    video_frames = read_video('input_videos/725edc7c-6e5e-fe5d-a1b4-219c3e07d32a_1280x720.mp4')

    # initialize tracker
    tracker = Tracker('models/best.pt')

    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/track_stubs2.pkl')

    # draw output
    ## draw object tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks)
    
    # save video
    save_video(output_video_frames, 'output_videos/output_video2.avi')

if __name__ == '__main__':
    main()
 